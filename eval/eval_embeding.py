"""比较 Milvus dense / sparse(BM25) / hybrid 裸检索，不经过 rerank 和 auto-merging。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "rag_v1.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
LEAF_LEVEL = 3


def load_dataset(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"数据集为空: {path}")
    for line_no, row in enumerate(rows, 1):
        if not row.get("question") or not row.get("gold_chunks"):
            raise ValueError(f"{path}:{line_no} 缺少 question 或 gold_chunks")
    return rows


def _document_id(chunk_id: str | None) -> str:
    return str(chunk_id or "").split("::", 1)[0]


def _first_rank(values: list[str], targets: set[str]) -> int | None:
    return next((index for index, value in enumerate(values, 1) if value in targets), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="比较 dense / sparse / hybrid 裸检索")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--modes", nargs="+", choices=["dense", "sparse", "hybrid"], default=["dense", "sparse", "hybrid"]
    )
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 8])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    from backend.vector.embedding import embedding_service
    from backend.vector.milvus_client import get_milvus_store

    store = get_milvus_store()
    if not store.has_collection():
        raise RuntimeError("Milvus 集合不存在")

    ks = sorted(set(args.k))
    stats = {mode: {k: defaultdict(float) for k in ks} for mode in args.modes}
    failures = defaultdict(int)
    latencies: dict[str, list[float]] = defaultdict(list)
    cases: list[dict] = []

    print(f"开始裸检索对比：{len(rows)} 条，modes={args.modes}")
    for index, row in enumerate(rows, 1):
        question = row["question"]
        gold_chunks = set(row["gold_chunks"])
        gold_docs = {_document_id(chunk_id) for chunk_id in gold_chunks}
        dense = None
        if any(mode in {"dense", "hybrid"} for mode in args.modes):
            dense = embedding_service.get_embeddings([question])[0]

        case = {"id": row.get("id"), "question": question, "modes": {}}
        for mode in args.modes:
            started = time.perf_counter()
            try:
                if mode == "dense":
                    hits = store.dense_retrieve(
                        dense_embedding=dense,
                        top_k=args.top_k,
                        filter_expr=f"chunk_level == {LEAF_LEVEL}",
                    )
                elif mode == "sparse":
                    hits = store.sparse_retrieve(
                        query=question,
                        top_k=args.top_k,
                        filter_expr=f"chunk_level == {LEAF_LEVEL}",
                    )
                else:
                    hits = store.hybrid_retrieve(
                        dense_embedding=dense,
                        query=question,
                        top_k=args.top_k,
                        filter_expr=f"chunk_level == {LEAF_LEVEL}",
                    )
            except Exception as exc:
                failures[mode] += 1
                case["modes"][mode] = {"status": "failed", "error": str(exc)}
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies[mode].append(elapsed_ms)
            returned = [str(hit.get("chunk_id") or "") for hit in hits]
            doc_rank = _first_rank([_document_id(value) for value in returned], gold_docs)
            evidence_rank = _first_rank(returned, gold_chunks)
            case["modes"][mode] = {
                "status": "ok",
                "document_rank": doc_rank,
                "evidence_rank": evidence_rank,
                "latency_ms": round(elapsed_ms, 1),
                "returned_chunk_ids": returned,
            }
            for k in ks:
                stats[mode][k]["document_hit"] += int(doc_rank is not None and doc_rank <= k)
                stats[mode][k]["evidence_hit"] += int(evidence_rank is not None and evidence_rank <= k)
                if doc_rank is not None and doc_rank <= k:
                    stats[mode][k]["document_mrr"] += 1 / doc_rank
        cases.append(case)
        if index % 5 == 0 or index == len(rows):
            print(f"  [{index}/{len(rows)}] 已完成")

    total = len(rows)
    summary = {
        "sample_count": total,
        "ks": ks,
        "modes": {
            mode: {
                "failed": failures[mode],
                "latency_mean_ms": sum(latencies[mode]) / len(latencies[mode]) if latencies[mode] else None,
                "metrics": {str(k): {metric: value / total for metric, value in stats[mode][k].items()} for k in ks},
            }
            for mode in args.modes
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"embedding-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "embedding_retrieval_comparison",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(args.dataset),
                "summary": summary,
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果已保存: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
