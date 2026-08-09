"""Embedding / 检索质量评估：接真实 Milvus 集合，对比 dense / sparse(BM25) / hybrid 三路裸检索。

两阶段 CLI：
  gen  从 level-3 叶子 chunk 抽样，用快模型生成"自然用户问法"并缓存真值（可复现）
  run  对缓存问题逐条跑检索，输出文档级 Hit@k / MRR@k 与严格 chunk 级命中对比表

运行前提：
  1. 基础设施已启动：`docker compose up -d`（Milvus 等）
  2. `.env` 配好 LLM_API_KEY / LLM_BASE_URL / LLM_FAST_MODEL（gen 阶段生成问题用）
  3. Milvus 集合已建且有 level-3 叶子 chunk（data/documents 上传过）

用法：
  uv run python eval/eval_embeding.py gen --count 30
  uv run python eval/eval_embeding.py run
  uv run python eval/eval_embeding.py run --modes dense sparse hybrid --k 1 3 5 --top-k 8
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录加入 sys.path，以便使用 backend 包导入
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "data" / "questions.json"
RESULTS_DIR = EVAL_DIR / "results"

# 检索/真值都限定在叶子层，与线上 retrieve_documents() 的 filter 一致
LEAF_LEVEL = 3

QUESTION_GEN_PROMPT = (
    "你是一位正在向知识库问答助手提问的用户。下面是文档中的一段内容：\n"
    "---\n{chunk_text}\n---\n"
    "请基于这段内容生成 1 条该用户可能提出的真实问题。要求：\n"
    "- 用自然、口语化的中文提问，不要照抄片段中的句子或关键词，尽量换一种说法；\n"
    "- 问题仅凭这段内容就能回答；\n"
    "- 只输出问题本身，不要任何解释、前缀或多余文字。"
)


# ---------- gen：生成并缓存问题真值 ----------

def _get_question_gen_model():
    from langchain.chat_models import init_chat_model

    model_name = os.getenv("LLM_FAST_MODEL") or os.getenv("LLM_MODEL")
    if not os.getenv("LLM_API_KEY") or not model_name:
        sys.exit("错误: 缺少 LLM_API_KEY / LLM_MODEL（或 LLM_FAST_MODEL），gen 阶段需要生成问题")
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.7,
    )


def _sample_leaf_chunks(store, count: int, seed: int) -> list[dict]:
    """按文档分组后轮流抽样，保证覆盖尽量多文档；seed 固定保证可复现。"""
    rng = random.Random(seed)
    chunks = store.query_all(
        filter_expr=f"chunk_level == {LEAF_LEVEL}",
        output_fields=["text", "filename", "page_number", "chunk_id", "root_chunk_id"],
    )
    chunks = [c for c in chunks if (c.get("text") or "").strip()]
    if not chunks:
        return []

    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c.get("filename", ""), []).append(c)
    docs = list(by_doc)
    rng.shuffle(docs)
    for key in docs:
        rng.shuffle(by_doc[key])

    sampled = []
    remaining = docs[:]
    idx = 0
    while len(sampled) < count and remaining:
        doc = remaining[idx % len(remaining)]
        sampled.append(by_doc[doc].pop())
        if not by_doc[doc]:
            remaining.pop(idx % len(remaining))
        else:
            idx += 1
    return sampled


def _generate_question(model, chunk_text: str) -> str:
    resp = model.invoke(
        [{"role": "user", "content": QUESTION_GEN_PROMPT.format(chunk_text=chunk_text[:1200])}]
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    return (content or "").strip()


def cmd_gen(args) -> None:
    from backend.vector.milvus_client import get_milvus_store

    if not args.regenerate and QUESTIONS_FILE.exists():
        print(f"questions.json 已存在: {QUESTIONS_FILE}（用 --regenerate 强制重新生成）")
        return

    store = get_milvus_store()
    if not store.has_collection():
        sys.exit("错误: Milvus 集合不存在，请先启动 docker compose up -d 并上传文档/初始化集合")

    sampled = _sample_leaf_chunks(store, args.count, args.seed)
    if not sampled:
        sys.exit(f"错误: 集合中没有 level-{LEAF_LEVEL} 的非空叶子 chunk，无法生成真值")

    print(f"抽样 {len(sampled)} 个叶子 chunk，开始生成问题（快模型，temperature=0.7）...")
    model = _get_question_gen_model()
    questions = []
    for i, c in enumerate(sampled, 1):
        try:
            question = _generate_question(model, c["text"])
        except Exception as e:
            print(f"  [{i}/{len(sampled)}] 生成失败，跳过: {e}")
            continue
        if not question:
            continue
        questions.append(
            {
                "question": question,
                "chunk_id": c.get("chunk_id", ""),
                "root_chunk_id": c.get("root_chunk_id", ""),
                "filename": c.get("filename", ""),
                "page_number": c.get("page_number", 0),
                "text": c.get("text", ""),
            }
        )

    if not questions:
        sys.exit("错误: 所有问题生成失败，请检查 LLM_API_KEY / LLM_BASE_URL / LLM_FAST_MODEL 配置")

    QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_FILE.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已缓存 {len(questions)} 条问题 -> {QUESTIONS_FILE}")


# ---------- run：跑检索并聚合指标 ----------

def _first_rank(hits: list[dict], field: str, target: str) -> int | None:
    """返回 top-k 中首个命中的 1-based 排名；无命中或目标为空返回 None。"""
    if not target:
        return None
    for i, hit in enumerate(hits, 1):
        if str(hit.get(field, "")) == target:
            return i
    return None


def _print_results(modes: list[str], ks: list[int], agg: dict, n: int, failed: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n检索质量评估  |  有效样本: {n}  检索失败: {failed}\n")
    table = Table(title="模式 × 指标")
    table.add_column("模式")
    for k in ks:
        table.add_column(f"Doc Hit@{k}")
    for k in ks:
        table.add_column(f"Doc MRR@{k}")
    for k in ks:
        table.add_column(f"Chunk Hit@{k}")
    for m in modes:
        row = [m]
        for k in ks:
            row.append(f"{agg[m][k]['doc_hit'] / n:.3f}" if n else "n/a")
        for k in ks:
            row.append(f"{agg[m][k]['mrr'] / n:.3f}" if n else "n/a")
        for k in ks:
            row.append(f"{agg[m][k]['chunk_hit'] / n:.3f}" if n else "n/a")
        table.add_row(*row)
    console.print(table)


def _save_results(modes: list[str], ks: list[int], agg: dict, per_question: list, failed: int) -> None:
    n = len(per_question)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": n,
        "failed": failed,
        "modes": {
            m: {
                str(k): {
                    key: (agg[m][k][key] / n if n else None)
                    for key in ("doc_hit", "mrr", "chunk_hit")
                }
                for k in ks
            }
            for m in modes
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(
        json.dumps({"summary": summary, "per_question": per_question}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已保存 -> {path}")


def cmd_run(args) -> None:
    if not QUESTIONS_FILE.exists():
        sys.exit(f"未找到 {QUESTIONS_FILE}，请先运行: uv run python eval/eval_embeding.py gen")
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    from backend.vector.embedding import embedding_service
    from backend.vector.milvus_client import get_milvus_store

    store = get_milvus_store()
    if not store.has_collection():
        sys.exit("错误: Milvus 集合不存在，请先启动 docker compose up -d 并初始化集合")

    modes, ks, top_k = args.modes, args.k, args.top_k
    filter_expr = f"chunk_level == {LEAF_LEVEL}"
    agg = {m: {k: {"doc_hit": 0, "mrr": 0.0, "chunk_hit": 0} for k in ks} for m in modes}

    failed = 0
    per_question = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        gt_doc = item.get("filename", "")
        gt_chunk = item.get("chunk_id", "")
        try:
            dense = embedding_service.get_embeddings([q])[0]
        except Exception as e:
            print(f"[{i}/{len(questions)}] embed 失败，跳过: {e}")
            failed += 1
            continue

        row = {"question": q, "gt_doc": gt_doc, "gt_chunk": gt_chunk, "modes": {}}
        for m in modes:
            try:
                if m == "dense":
                    hits = store.dense_retrieve(dense_embedding=dense, top_k=top_k, filter_expr=filter_expr)
                elif m == "sparse":
                    hits = store.sparse_retrieve(query=q, top_k=top_k, filter_expr=filter_expr)
                else:  # hybrid
                    hits = store.hybrid_retrieve(
                        dense_embedding=dense, query=q, top_k=top_k, filter_expr=filter_expr
                    )
            except Exception as e:
                print(f"[{i}/{len(questions)}] 模式 {m} 检索失败，跳过: {e}")
                failed += 1
                continue

            doc_rank = _first_rank(hits, "filename", gt_doc)
            chunk_rank = _first_rank(hits, "chunk_id", gt_chunk)
            row["modes"][m] = {"doc_rank": doc_rank, "chunk_rank": chunk_rank}
            for k in ks:
                if doc_rank is not None and doc_rank <= k:
                    agg[m][k]["doc_hit"] += 1
                    agg[m][k]["mrr"] += 1.0 / doc_rank
                if chunk_rank is not None and chunk_rank <= k:
                    agg[m][k]["chunk_hit"] += 1
        per_question.append(row)

    _print_results(modes, ks, agg, len(per_question), failed)
    _save_results(modes, ks, agg, per_question, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding / 检索质量评估（接真实 Milvus）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen", help="从叶子 chunk 抽样并用快模型生成问题真值")
    p_gen.add_argument("--count", type=int, default=30, help="生成的问题数量（默认 30）")
    p_gen.add_argument("--seed", type=int, default=42, help="抽样随机种子（默认 42）")
    p_gen.add_argument("--regenerate", action="store_true", help="强制重新生成，忽略缓存")
    p_gen.set_defaults(func=cmd_gen)

    p_run = sub.add_parser("run", help="对缓存问题跑检索并输出指标")
    p_run.add_argument(
        "--modes", nargs="+", choices=["dense", "sparse", "hybrid"], default=["hybrid"]
    )
    p_run.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    p_run.add_argument("--top-k", type=int, default=8)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
