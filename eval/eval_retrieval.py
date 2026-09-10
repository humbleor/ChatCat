"""端到端检索质量评测：直接调后端 retrieve_documents() 跑真实 RAG 检索链路。

目标指标：
  - Doc Hit@K / Chunk Hit@K / Doc MRR@K
  - 检索延迟 P50/P95
  - Reranker 启用/应用/失败率
  - Auto-merging 替换次数与启用率

运行前提：
  1. docker compose up -d（Postgres / Redis / Milvus）
  2. Milvus 集合里已上传过文档（有 level-3 叶子 chunk）
  3. `eval/eval_embeding.py gen` 已生成 questions.json

用法：
  uv run python eval/eval_retrieval.py
  uv run python eval/eval_retrieval.py --k 1 3 5 --top-k 8
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "data" / "questions.json"
RESULTS_DIR = EVAL_DIR / "results"


def _first_rank(hits: list[dict], field: str, target: str) -> int | None:
    if not target:
        return None
    for i, hit in enumerate(hits, 1):
        if str(hit.get(field, "")) == target:
            return i
    return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _print_report(summary: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(
        f"\n检索质量评测  |  样本: {summary['sample_count']}  检索失败: {summary['failed']}  "
        f"空结果: {summary['empty_hits']}\n"
    )

    ks = summary["ks"]
    table = Table(title="召回指标")
    table.add_column("指标")
    for k in ks:
        table.add_column(f"@{k}")
    for k in ks:
        table.add_column(f"@{k}")
    for k in ks:
        table.add_column(f"@{k}")

    def _row(name: str, src: dict) -> list:
        return [name] + [f"{src[k]:.3f}" for k in ks]

    table.add_row(*_row("Doc Hit", summary["doc_hit"]))
    table.add_row(*_row("Root-Chunk Hit", summary["root_chunk_hit"]))
    table.add_row(*_row("Chunk Hit", summary["chunk_hit"]))
    table.add_row(*_row("Doc MRR", summary["doc_mrr"]))
    console.print(table)

    perf = Table(title="性能与开关")
    perf.add_column("指标")
    perf.add_column("值")
    p50 = summary["latency_ms"]["p50"]
    p95 = summary["latency_ms"]["p95"]
    mean = summary["latency_ms"]["mean"]
    perf.add_row("检索 P50 (ms)", f"{p50:.1f}" if p50 is not None else "-")
    perf.add_row("检索 P95 (ms)", f"{p95:.1f}" if p95 is not None else "-")
    perf.add_row("检索均值 (ms)", f"{mean:.1f}" if mean is not None else "-")
    perf.add_row("Rerank 应用率", f"{summary['rerank']['applied_rate']:.3f}（{summary['rerank']['applied']}/{summary['rerank']['seen']}）")
    perf.add_row("Rerank 失败次数", f"{summary['rerank']['failed']}")
    perf.add_row("Auto-merge 替换总块数", f"{summary['auto_merge']['replaced_total']}")
    perf.add_row("Auto-merge 实际启用率", f"{summary['auto_merge']['applied_rate']:.3f}（{summary['auto_merge']['applied']}/{summary['auto_merge']['seen']}）")
    perf.add_row(
        "检索模式 (hybrid/dense_fallback/failed)",
        f"{summary['retrieval_mode']['hybrid']} / {summary['retrieval_mode']['dense_fallback']} / {summary['retrieval_mode']['failed']}",
    )
    console.print(perf)


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端检索质量评测（接真实 RAG 检索链路）")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5], help="Hit@K / MRR@K 的 K 值")
    parser.add_argument("--top-k", type=int, default=5, help="retrieve_documents 的 top_k")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条样本（0 表示全部）")
    parser.add_argument("--retries", type=int, default=1, help="单条样本失败重试次数")
    parser.add_argument("--warmup", type=int, default=3, help="不计入指标的预热查询数（0 关闭）")
    args = parser.parse_args()

    if not QUESTIONS_FILE.exists():
        sys.exit(f"未找到 {QUESTIONS_FILE}，请先运行: uv run python eval/eval_embeding.py gen")

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.limit > 0:
        questions = questions[: args.limit]

    from backend.rag.rag_utils import retrieve_documents  # noqa: E402

    if args.warmup > 0:
        print(f"预热：跑 {args.warmup} 条不计分（让 BGE-M3 / Rerank 端完成首次加载）...")
        for j in range(args.warmup):
            try:
                retrieve_documents(f"warmup query {j}", top_k=args.top_k)
            except Exception as e:
                print(f"  预热 {j + 1} 失败（忽略）: {e}")
        print("预热完成。\n")

    ks = sorted(set(args.k))
    agg = {k: {"doc_hit": 0, "doc_mrr": 0.0, "chunk_hit": 0, "root_chunk_hit": 0} for k in ks}
    failed = 0
    empty_hits = 0
    latencies: list[float] = []
    rerank_applied = 0
    rerank_failed = 0
    rerank_seen = 0
    auto_merge_replaced = 0
    auto_merge_applied = 0
    auto_merge_seen = 0
    retrieval_mode_counts = {"hybrid": 0, "dense_fallback": 0, "failed": 0}
    per_question: list[dict] = []

    print(f"开始评测：{len(questions)} 条样本，top_k={args.top_k}，ks={ks}")
    for i, item in enumerate(questions, 1):
        q = item["question"]
        gt_doc = item.get("filename", "")
        gt_chunk = item.get("chunk_id", "")
        gt_root = item.get("root_chunk_id", "")
        last_err = None
        hits: list[dict] = []
        meta: dict = {}
        for attempt in range(args.retries + 1):
            t0 = time.perf_counter()
            try:
                result = retrieve_documents(q, top_k=args.top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                hits = result.get("docs", []) if isinstance(result, dict) else []
                meta = result.get("meta", {}) if isinstance(result, dict) else {}
                break
            except Exception as e:
                last_err = str(e)
                elapsed_ms = 0.0
        else:
            failed += 1
            print(f"[{i}/{len(questions)}] 检索失败 {args.retries + 1} 次，跳过: {last_err}")
            per_question.append({"question": q, "gt_doc": gt_doc, "gt_chunk": gt_chunk, "error": last_err})
            continue

        if not hits:
            empty_hits += 1
        latencies.append(elapsed_ms)

        doc_rank = _first_rank(hits, "filename", gt_doc)
        chunk_rank = _first_rank(hits, "chunk_id", gt_chunk)
        # 根块级命中：先按 root_chunk_id 精确匹配；缺失时退化为文档级（保留有效样本）
        if gt_root:
            root_rank = _first_rank(hits, "root_chunk_id", gt_root)
            if root_rank is None and gt_doc:
                root_rank = doc_rank  # 回退到 doc_rank，避免 gt_root 缺失把所有样本判为未命中
        else:
            root_rank = doc_rank
        for k in ks:
            if doc_rank is not None and doc_rank <= k:
                agg[k]["doc_hit"] += 1
                agg[k]["doc_mrr"] += 1.0 / doc_rank
            if chunk_rank is not None and chunk_rank <= k:
                agg[k]["chunk_hit"] += 1
            if root_rank is not None and root_rank <= k:
                agg[k]["root_chunk_hit"] += 1

        if meta:
            if meta.get("rerank_enabled"):
                rerank_seen += 1
                if meta.get("rerank_applied"):
                    rerank_applied += 1
                if meta.get("rerank_error"):
                    rerank_failed += 1
            if meta.get("auto_merge_enabled"):
                auto_merge_seen += 1
                if meta.get("auto_merge_applied"):
                    auto_merge_applied += 1
                auto_merge_replaced += int(meta.get("auto_merge_replaced_chunks", 0) or 0)
            mode = meta.get("retrieval_mode")
            if mode in retrieval_mode_counts:
                retrieval_mode_counts[mode] += 1

        per_question.append(
            {
                "question": q,
                "gt_doc": gt_doc,
                "gt_chunk": gt_chunk,
                "gt_root_chunk_id": gt_root,
                "latency_ms": round(elapsed_ms, 1),
                "doc_rank": doc_rank,
                "chunk_rank": chunk_rank,
                "root_rank": root_rank,
                "hit_count": len(hits),
                "retrieval_mode": meta.get("retrieval_mode"),
                "rerank_enabled": meta.get("rerank_enabled"),
                "rerank_applied": meta.get("rerank_applied"),
                "rerank_error": meta.get("rerank_error"),
                "auto_merge_applied": meta.get("auto_merge_applied"),
                "auto_merge_replaced_chunks": meta.get("auto_merge_replaced_chunks"),
            }
        )

        if i % 5 == 0 or i == len(questions):
            print(f"  [{i}/{len(questions)}] 已完成")

    n = len(per_question)
    if n == 0:
        sys.exit("没有任何样本产出结果")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": n,
        "failed": failed,
        "empty_hits": empty_hits,
        "ks": ks,
        "doc_hit": {k: (agg[k]["doc_hit"] / n) for k in ks},
        "doc_mrr": {k: (agg[k]["doc_mrr"] / n) for k in ks},
        "chunk_hit": {k: (agg[k]["chunk_hit"] / n) for k in ks},
        "root_chunk_hit": {k: (agg[k]["root_chunk_hit"] / n) for k in ks},
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
        },
        "rerank": {
            "seen": rerank_seen,
            "applied": rerank_applied,
            "applied_rate": (rerank_applied / rerank_seen) if rerank_seen else 0.0,
            "failed": rerank_failed,
        },
        "auto_merge": {
            "seen": auto_merge_seen,
            "applied": auto_merge_applied,
            "applied_rate": (auto_merge_applied / auto_merge_seen) if auto_merge_seen else 0.0,
            "replaced_total": auto_merge_replaced,
        },
        "retrieval_mode": retrieval_mode_counts,
    }

    _print_report(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"retrieval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(
        json.dumps({"summary": summary, "per_question": per_question}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已保存 -> {path}")


if __name__ == "__main__":
    main()
