"""用版本化 JSONL 数据集评测 ChatCat 的真实检索链路，并支持 baseline 回归门禁。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "rag_v1.jsonl"
DEFAULT_BASELINE = EVAL_DIR / "baselines" / "rag_v1-retrieval.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_dataset(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"数据集为空: {path}")
    required = {"id", "question", "gold_answer", "gold_chunks", "tags"}
    seen: set[str] = set()
    for line_no, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_no} 缺少字段: {sorted(missing)}")
        if row["id"] in seen:
            raise ValueError(f"{path}:{line_no} 重复 id: {row['id']}")
        if not row["gold_chunks"]:
            raise ValueError(f"{path}:{line_no} gold_chunks 不能为空")
        seen.add(row["id"])
    return rows, hashlib.sha256(raw).hexdigest()


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _document_id(chunk_id: str | None) -> str:
    return str(chunk_id or "").split("::", 1)[0]


def _first_rank(values: list[str], targets: set[str]) -> int | None:
    return next((index for index, value in enumerate(values, 1) if value in targets), None)


def _load_gold_families(rows: list[dict]) -> dict[str, dict[str, str]]:
    """读取当前 Milvus 元数据，用 parent/root 关系容忍 auto-merging 返回父块。"""
    from backend.vector.milvus_client import get_milvus_store

    wanted = {chunk_id for row in rows for chunk_id in row["gold_chunks"]}
    records = get_milvus_store().query_all(
        filter_expr="chunk_level == 3",
        output_fields=["chunk_id", "parent_chunk_id", "root_chunk_id"],
    )
    found = {
        record["chunk_id"]: {
            "parent_chunk_id": str(record.get("parent_chunk_id") or ""),
            "root_chunk_id": str(record.get("root_chunk_id") or ""),
        }
        for record in records
        if record.get("chunk_id") in wanted
    }
    missing = sorted(wanted - found.keys())
    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"{len(missing)} 个 gold chunk 不在当前 Milvus 中:\n{preview}")
    return found


def _evidence_targets(gold_chunks: list[str], families: dict[str, dict[str, str]]) -> set[str]:
    targets = set(gold_chunks)
    for chunk_id in gold_chunks:
        targets.update(value for value in families[chunk_id].values() if value)
    return targets


def _metric_at(summary: dict, name: str, k: int) -> float | None:
    value = summary.get(name, {}).get(str(k))
    return float(value) if value is not None else None


def compare_baseline(
    summary: dict,
    baseline: dict,
    *,
    max_hit_regression: float,
    max_failure_increase: float,
    max_latency_increase: float,
) -> list[dict]:
    old = baseline["summary"]
    checks: list[dict] = []
    for name in ("document_hit", "evidence_hit"):
        for k in summary["ks"]:
            current = _metric_at(summary, name, k)
            previous = _metric_at(old, name, k)
            if current is None or previous is None:
                continue
            minimum = max(0.0, previous - max_hit_regression)
            checks.append(
                {
                    "metric": f"{name}@{k}",
                    "baseline": previous,
                    "current": current,
                    "threshold": minimum,
                    "passed": current >= minimum,
                }
            )

    current_failure = float(summary["failure_rate"])
    old_failure = float(old["failure_rate"])
    checks.append(
        {
            "metric": "failure_rate",
            "baseline": old_failure,
            "current": current_failure,
            "threshold": old_failure + max_failure_increase,
            "passed": current_failure <= old_failure + max_failure_increase,
        }
    )
    current_rerank_failure = float(summary["rerank"]["failure_rate"])
    old_rerank_failure = float(old["rerank"].get("failure_rate", 0.0))
    checks.append(
        {
            "metric": "rerank_failure_rate",
            "baseline": old_rerank_failure,
            "current": current_rerank_failure,
            "threshold": old_rerank_failure + max_failure_increase,
            "passed": current_rerank_failure <= old_rerank_failure + max_failure_increase,
        }
    )
    current_p95 = summary["latency_ms"]["p95"]
    old_p95 = old["latency_ms"]["p95"]
    if current_p95 is not None and old_p95 is not None:
        checks.append(
            {
                "metric": "latency_p95_ms",
                "baseline": old_p95,
                "current": current_p95,
                "threshold": old_p95 * (1 + max_latency_increase),
                "passed": current_p95 <= old_p95 * (1 + max_latency_increase),
            }
        )
    return checks


def _print_report(summary: dict, gates: list[dict] | None = None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(
        f"\n检索质量评测 | 样本 {summary['sample_count']} | 失败 {summary['failed']} | 空结果 {summary['empty_hits']}\n"
    )
    table = Table(title="召回指标")
    table.add_column("指标")
    for k in summary["ks"]:
        table.add_column(f"@{k}")
    for label, key in (
        ("Document Hit", "document_hit"),
        ("Evidence Hit", "evidence_hit"),
        ("Evidence Coverage", "evidence_coverage"),
        ("Document MRR", "document_mrr"),
    ):
        table.add_row(label, *[f"{summary[key][str(k)]:.3f}" for k in summary["ks"]])
    console.print(table)

    perf = Table(title="性能与运行状态")
    perf.add_column("指标")
    perf.add_column("值")
    for key in ("mean", "p50", "p95"):
        value = summary["latency_ms"][key]
        perf.add_row(f"延迟 {key}", f"{value:.1f} ms" if value is not None else "-")
    perf.add_row("失败率", f"{summary['failure_rate']:.3f}")
    perf.add_row("Rerank 失败", str(summary["rerank"]["failed"]))
    perf.add_row("Dense fallback", str(summary["retrieval_mode"]["dense_fallback"]))
    console.print(perf)

    if gates is not None:
        gate_table = Table(title="Baseline Gate")
        for column in ("状态", "指标", "当前", "门槛", "Baseline"):
            gate_table.add_column(column)
        for gate in gates:
            gate_table.add_row(
                "PASS" if gate["passed"] else "FAIL",
                gate["metric"],
                f"{gate['current']:.4f}",
                f"{gate['threshold']:.4f}",
                f"{gate['baseline']:.4f}",
            )
        console.print(gate_table)


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 rag_v1.jsonl 评测真实 RAG 检索链路")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--allow-degraded-baseline", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 8])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-hit-regression", type=float, default=0.03)
    parser.add_argument("--max-failure-increase", type=float, default=0.0)
    parser.add_argument("--max-latency-increase", type=float, default=0.25)
    args = parser.parse_args()

    rows, dataset_sha256 = load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    families = _load_gold_families(rows)

    from backend.rag.rag_utils import retrieve_documents

    for index in range(args.warmup):
        try:
            retrieve_documents(f"warmup query {index}", top_k=args.top_k)
        except Exception as exc:
            print(f"预热失败（忽略）: {exc}")

    ks = sorted(set(args.k))
    counters = {k: {"document_hit": 0, "evidence_hit": 0, "evidence_coverage": 0.0, "document_mrr": 0.0} for k in ks}
    failed = 0
    empty_hits = 0
    latencies: list[float] = []
    rerank = Counter(seen=0, applied=0, failed=0)
    auto_merge = Counter(seen=0, applied=0, replaced_total=0)
    modes = Counter(hybrid=0, dense_fallback=0, failed=0)
    by_tag: dict[str, Counter] = defaultdict(Counter)
    details: list[dict[str, Any]] = []

    print(f"开始评测 {len(rows)} 条样本，dataset_sha256={dataset_sha256[:12]}...")
    for index, row in enumerate(rows, 1):
        last_error = ""
        result: dict = {}
        started = time.perf_counter()
        for _attempt in range(args.retries + 1):
            try:
                result = retrieve_documents(row["question"], top_k=args.top_k)
                last_error = ""
                break
            except Exception as exc:
                last_error = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if last_error:
            failed += 1
            details.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "tags": row["tags"],
                    "status": "failed",
                    "failure_stage": "retrieval",
                    "error": last_error,
                    "latency_ms": round(elapsed_ms, 1),
                }
            )
            continue

        docs = result.get("docs", [])
        meta = result.get("meta", {})
        if not docs:
            empty_hits += 1
        latencies.append(elapsed_ms)

        returned_ids = [str(doc.get("chunk_id") or "") for doc in docs]
        returned_docs = [_document_id(chunk_id) for chunk_id in returned_ids]
        gold_chunks = set(row["gold_chunks"])
        gold_docs = {_document_id(chunk_id) for chunk_id in gold_chunks}
        family_targets = _evidence_targets(row["gold_chunks"], families)
        returned_families = [
            {
                str(doc.get("chunk_id") or ""),
                str(doc.get("parent_chunk_id") or ""),
                str(doc.get("root_chunk_id") or ""),
            }
            for doc in docs
        ]
        document_rank = _first_rank(returned_docs, gold_docs)
        evidence_rank = next(
            (rank for rank, values in enumerate(returned_families, 1) if values & family_targets),
            None,
        )

        per_k: dict[str, dict] = {}
        for k in ks:
            doc_hit = document_rank is not None and document_rank <= k
            evidence_hit = evidence_rank is not None and evidence_rank <= k
            covered = {
                gold
                for gold in gold_chunks
                if any(
                    {
                        str(doc.get("chunk_id") or ""),
                        str(doc.get("parent_chunk_id") or ""),
                        str(doc.get("root_chunk_id") or ""),
                    }
                    & _evidence_targets([gold], families)
                    for doc in docs[:k]
                )
            }
            coverage = len(covered) / len(gold_chunks)
            counters[k]["document_hit"] += int(doc_hit)
            counters[k]["evidence_hit"] += int(evidence_hit)
            counters[k]["evidence_coverage"] += coverage
            if doc_hit:
                counters[k]["document_mrr"] += 1 / document_rank
            per_k[str(k)] = {
                "document_hit": doc_hit,
                "evidence_hit": evidence_hit,
                "evidence_coverage": coverage,
            }
            for tag in row["tags"]:
                by_tag[tag][f"count@{k}"] += 1
                by_tag[tag][f"document_hit@{k}"] += int(doc_hit)
                by_tag[tag][f"evidence_hit@{k}"] += int(evidence_hit)

        if meta.get("rerank_enabled"):
            rerank["seen"] += 1
            rerank["applied"] += int(bool(meta.get("rerank_applied")))
            rerank["failed"] += int(bool(meta.get("rerank_error")))
        if meta.get("auto_merge_enabled"):
            auto_merge["seen"] += 1
            auto_merge["applied"] += int(bool(meta.get("auto_merge_applied")))
            auto_merge["replaced_total"] += int(meta.get("auto_merge_replaced_chunks", 0) or 0)
        modes[str(meta.get("retrieval_mode") or "unknown")] += 1
        details.append(
            {
                "id": row["id"],
                "question": row["question"],
                "tags": row["tags"],
                "status": "ok",
                "latency_ms": round(elapsed_ms, 1),
                "document_rank": document_rank,
                "evidence_rank": evidence_rank,
                "metrics": per_k,
                "returned_chunk_ids": returned_ids,
                "retrieval_meta": meta,
            }
        )
        if index % 5 == 0 or index == len(rows):
            print(f"  [{index}/{len(rows)}] 已完成")

    total = len(rows)
    summary = {
        "sample_count": total,
        "failed": failed,
        "failure_rate": failed / total,
        "empty_hits": empty_hits,
        "ks": ks,
        "document_hit": {str(k): counters[k]["document_hit"] / total for k in ks},
        "evidence_hit": {str(k): counters[k]["evidence_hit"] / total for k in ks},
        "evidence_coverage": {str(k): counters[k]["evidence_coverage"] / total for k in ks},
        "document_mrr": {str(k): counters[k]["document_mrr"] / total for k in ks},
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "rerank": {
            **rerank,
            "applied_rate": rerank["applied"] / rerank["seen"] if rerank["seen"] else 0.0,
        },
        "auto_merge": {
            **auto_merge,
            "applied_rate": auto_merge["applied"] / auto_merge["seen"] if auto_merge["seen"] else 0.0,
        },
        "retrieval_mode": dict(modes),
        "by_tag": {
            tag: {
                key: (
                    value / values[key.replace("document_hit", "count").replace("evidence_hit", "count")]
                    if (
                        "_hit@" in key and values[key.replace("document_hit", "count").replace("evidence_hit", "count")]
                    )
                    else value
                )
                for key, value in values.items()
            }
            for tag, values in by_tag.items()
        },
    }
    report = {
        "schema_version": 1,
        "kind": "rag_retrieval_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "config": {"top_k": args.top_k, "ks": ks, "retries": args.retries},
        "summary": summary,
        "cases": details,
    }

    gates: list[dict] | None = None
    if args.update_baseline:
        if args.limit:
            raise ValueError("--update-baseline 不能与 --limit 一起使用")
        if not args.allow_degraded_baseline and (failed or rerank["failed"]):
            raise RuntimeError("存在 retrieval/rerank Provider 失败，拒绝更新 baseline")
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Baseline 已更新: {args.baseline}")
    elif args.baseline.exists() and not args.no_gate:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        gates = compare_baseline(
            summary,
            baseline,
            max_hit_regression=args.max_hit_regression,
            max_failure_increase=args.max_failure_increase,
            max_latency_increase=args.max_latency_increase,
        )
    elif not args.no_gate:
        print(f"尚无 baseline：{args.baseline}；本次只生成结果，不执行 gate。")

    _print_report(summary, gates)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"retrieval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    output.write_text(
        json.dumps({**report, "gates": gates}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已保存: {output}")
    if gates and any(not gate["passed"] for gate in gates):
        print("Baseline gate 失败。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
