"""Embedding / 检索质量评估：接真实 Milvus 集合，对比 dense / sparse(BM25) / hybrid 三路裸检索。

两阶段 CLI：
  gen  从 level-3 叶子 chunk 抽样，用本地模板拼出"自然问句"并缓存真值（可复现、不调 LLM）
  run  对缓存问题逐条跑检索，输出文档级 Hit@k / MRR@k 与严格 chunk 级命中对比表

运行前提：
  1. 基础设施已启动：`docker compose up -d`（Milvus 等）
  2. Milvus 集合已建且有 level-3 叶子 chunk（data/documents 上传过）

用法：
  uv run python eval/eval_embeding.py gen --count 30
  uv run python eval/eval_embeding.py run
  uv run python eval/eval_embeding.py run --modes dense sparse hybrid --k 1 3 5 --top-k 8
"""

import argparse
import json
import os
import random
import re
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

# 模板式问题生成：完全不调 LLM，保证可复现、零 token。
# 思路：从 chunk 文本里抽 1~2 个关键短语，按句式模板拼成单行中文问句。
_TEMPLATES = [
    "这段内容里提到的{k}是什么？",
    "{k}在文档里是怎么描述的？",
    "什么是{k}？它有什么特点？",
    "请解释一下{k}的含义。",
    "{a}和{b}有什么区别？",
    "{file}里关于{k}的说明是什么？",
]
_STOPWORDS = {
    "的", "是", "在", "和", "了", "与", "及", "或", "等", "我们", "你", "它", "他", "她",
    "这", "那", "一个", "一些", "可以", "通过", "对", "为", "以", "上", "下", "中", "等",
    "from", "the", "and", "of", "in", "to", "for", "with", "a", "an", "is", "are", "by",
    "图", "表", "如下", "所示", "分别", "其中", "包括", "使用", "采用", "需要", "本文",
    "实验", "结果", "分析", "方法", "数据", "模型", "本文", "本节", "工作",
}
_KEYWORD_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_\-]{2,}"     # 英文术语/缩写
    r"|[一-龥]{2,8}",                  # 中文 2~8 字短语
)


def _extract_keywords(text: str, limit: int = 3) -> list[str]:
    """从 chunk 文本里挑 1~limit 个关键词：英文术语优先，其次中文短语，按出现频次/长度排序。"""
    from collections import Counter

    text = (text or "").strip()
    if not text:
        return []
    counter: Counter = Counter()
    for m in _KEYWORD_RE.finditer(text):
        kw = m.group(0).strip()
        if not kw or kw.lower() in _STOPWORDS:
            continue
        if len(kw) == 1 and kw not in {"A", "I"}:
            continue
        counter[kw] += 1
    # 优先按出现次数，再按长度；英文术语靠前
    sorted_kws = sorted(
        counter.items(),
        key=lambda kv: (-kv[1], -len(kv[0]), kv[0]),
    )
    return [kw for kw, _ in sorted_kws[:limit]]


def _file_stem(filename: str) -> str:
    """从文件名里抽清爽短名（去扩展名、连字符等）。"""
    stem = (filename or "").strip()
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    return stem.strip()[:24]


def _template_question(text: str, filename: str, rng: random.Random) -> str:
    """根据 chunk 文本和文件名用模板拼一行问句。"""
    kws = _extract_keywords(text)
    if not kws:
        # 文本无可用短语时用文件名兜底
        kws = [_file_stem(filename)] or ["这段内容"]
    rng.shuffle(kws)
    head = kws[0]
    tmpl = rng.choice(_TEMPLATES)
    if "{a}" in tmpl and "{b}" in tmpl and len(kws) >= 2:
        return tmpl.format(a=kws[0], b=kws[1], k=head, file=_file_stem(filename))
    return tmpl.format(k=head, file=_file_stem(filename))


# ---------- gen：生成并缓存问题真值 ----------

def _get_question_gen_model():
    # 保留接口以兼容旧用法，但模板生成模式下不会调用模型
    from langchain.chat_models import init_chat_model

    model_name = os.getenv("LLM_FAST_MODEL") or os.getenv("LLM_MODEL")
    if not os.getenv("LLM_API_KEY") or not model_name:
        return None
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


def _audit_questions(questions: list) -> None:
    """生成后轻量审计：覆盖文档数、长度、是否含 <think> 残留、是否过短。"""
    from collections import Counter

    if not questions:
        print("⚠️ 审计: 没有可用的样本")
        return
    files = Counter(item.get("filename", "") for item in questions)
    lengths = [len(item["question"]) for item in questions]
    short = [item for item in questions if len(item["question"]) < 6]
    think_leak = [item for item in questions if "<think>" in item["question"].lower()]
    print("\n[样本审计]")
    print(f"  总样本: {len(questions)}")
    print(f"  覆盖文档数: {len(files)}")
    print(f"  问题长度 (min/median/max): {min(lengths)} / {sorted(lengths)[len(lengths)//2]} / {max(lengths)}")
    if short:
        print(f"  ⚠️ 过短问题: {len(short)} 条（<6 字）")
    if think_leak:
        print(f"  ⚠️ 仍含 <think>: {len(think_leak)} 条")
    top_files = files.most_common(5)
    print("  覆盖 top 文档:")
    for fn, n in top_files:
        print(f"    - {fn}: {n}")


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

    print(f"抽样 {len(sampled)} 个叶子 chunk，开始按模板拼装问题（不调 LLM，固定 seed={args.seed}）...")
    rng = random.Random(args.seed + 1)
    questions = []
    for i, c in enumerate(sampled, 1):
        text = c.get("text", "") or ""
        filename = c.get("filename", "")
        if not text.strip():
            print(f"  [{i}/{len(sampled)}] 文本为空，跳过")
            continue
        question = _template_question(text, filename, rng)
        if not question:
            print(f"  [{i}/{len(sampled)}] 模板生成失败，跳过")
            continue
        questions.append(
            {
                "question": question,
                "chunk_id": c.get("chunk_id", ""),
                "root_chunk_id": c.get("root_chunk_id", ""),
                "filename": filename,
                "page_number": c.get("page_number", 0),
                "text": text,
            }
        )

    if not questions:
        sys.exit("错误: 没有任何 chunk 能拼出可用问题")

    _audit_questions(questions)

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
