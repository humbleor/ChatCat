"""完整 RAG 回答评测：独立 ChatRun、结构化 LLM Judge、baseline 回归门禁。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "rag_v1.jsonl"
DEFAULT_BASELINE = EVAL_DIR / "baselines" / "rag_v1-answer.json"
RESULTS_DIR = EVAL_DIR / "results"
METRICS = ("groundedness", "correctness", "relevance", "completeness")


def load_dataset(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"数据集为空: {path}")
    return rows, hashlib.sha256(raw).hexdigest()


def ensure_eval_user(username: str) -> None:
    from backend.infra.auth import get_password_hash
    from backend.infra.database import SessionLocal
    from backend.models.models import User

    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == username).first():
            db.add(User(username=username, password_hash=get_password_hash(uuid4().hex), role="user"))
            db.commit()
    finally:
        db.close()


def _judge_model():
    from langchain.chat_models import init_chat_model

    model = os.getenv("LLM_JUDGE_MODEL") or os.getenv("LLM_FAST_MODEL") or os.getenv("LLM_MODEL")
    if not model or not os.getenv("LLM_API_KEY"):
        raise RuntimeError("需要配置 LLM_API_KEY 和 LLM_JUDGE_MODEL/LLM_FAST_MODEL/LLM_MODEL")
    return init_chat_model(
        model=model,
        model_provider="openai",
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    ), model


def _evidence_text(rag_trace: dict) -> str:
    chunks = rag_trace.get("retrieved_chunks") or []
    parts = []
    for index, chunk in enumerate(chunks, 1):
        text = str(chunk.get("text") or "")
        parts.append(f"[{index}] {chunk.get('chunk_id', '')}\n{text[:2400]}")
    return "\n\n".join(parts)[:12000]


def judge_answer(model, sample: dict, answer: str, rag_trace: dict) -> dict:
    prompt = f"""你是严格的 RAG 评测器。只根据参考答案和检索证据评分，不使用外部知识。
所有 0-1 分数使用 0.0、0.25、0.5、0.75、1.0。
groundedness: 回答中的主张是否得到检索证据支持。
correctness: 回答与参考答案是否事实一致。
relevance: 是否直接回答问题且没有明显跑题。
completeness: 是否覆盖参考答案的关键点。
unsupported_claims: 回答中无法由证据支持的具体主张数量。
只返回 JSON 对象，不要 Markdown：
{{"groundedness":0.0,"correctness":0.0,"relevance":0.0,"completeness":0.0,
"unsupported_claims":0,"reason":"不超过120字"}}

问题：
{sample["question"]}

参考答案：
{sample["gold_answer"]}

系统回答：
{answer}

检索证据：
{_evidence_text(rag_trace)}
"""
    raw = str(model.invoke(prompt).content)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Judge 未返回 JSON: {raw[:200]}")
    result = json.loads(match.group(0))
    for metric in METRICS:
        result[metric] = min(1.0, max(0.0, float(result[metric])))
    result["unsupported_claims"] = max(0, int(result.get("unsupported_claims", 0)))
    return result


def compare_baseline(summary: dict, baseline: dict, tolerance: float) -> list[dict]:
    old = baseline["summary"]
    gates = []
    for metric in METRICS:
        current = summary["judge_metrics"].get(metric)
        previous = old["judge_metrics"].get(metric)
        if current is None or previous is None:
            continue
        threshold = max(0.0, previous - tolerance)
        gates.append(
            {
                "metric": metric,
                "baseline": previous,
                "current": current,
                "threshold": threshold,
                "passed": current >= threshold,
            }
        )
    for metric in ("generation_failure_rate", "judge_failure_rate", "empty_answer_rate"):
        current = summary[metric]
        previous = old[metric]
        threshold = previous
        gates.append(
            {
                "metric": metric,
                "baseline": previous,
                "current": current,
                "threshold": threshold,
                "passed": current <= threshold,
            }
        )
    return gates


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 rag_v1.jsonl 评测完整 ChatRun 回答")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--username", default="rag_eval")
    parser.add_argument("--max-score-regression", type=float, default=0.03)
    args = parser.parse_args()

    rows, dataset_sha256 = load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    from backend.agent.agent import chat_with_agent
    from backend.agent.run_manager import mark_failed, reserve_run
    from backend.infra.checkpointer import init_checkpointer
    from backend.infra.database import init_db

    init_db()
    init_checkpointer()
    ensure_eval_user(args.username)
    judge = None
    judge_model_name = None
    if not args.no_judge:
        judge, judge_model_name = _judge_model()

    cases = []
    scores: dict[str, list[float]] = defaultdict(list)
    generation_failed = 0
    judge_failed = 0
    empty_answers = 0
    started_run = datetime.now(timezone.utc)

    for index, sample in enumerate(rows, 1):
        session_id = f"eval_{sample['id']}_{uuid4().hex[:12]}"
        request_id = f"eval-{dataset_sha256[:12]}-{sample['id']}-{uuid4().hex[:8]}"
        reservation = reserve_run(
            username=args.username,
            session_id=session_id,
            message=sample["question"],
            request_id=request_id,
        )
        started = time.perf_counter()
        try:
            result = chat_with_agent(
                user_text=sample["question"],
                user_id=args.username,
                session_id=session_id,
                run_id=reservation.run_id,
            )
        except Exception as exc:
            mark_failed(reservation.run_id, str(exc), code="EVAL_GENERATION_FAILED")
            generation_failed += 1
            cases.append(
                {
                    "id": sample["id"],
                    "status": "failed",
                    "failure_stage": "generation",
                    "error": str(exc),
                    "run_id": reservation.run_id,
                }
            )
            continue

        elapsed_ms = (time.perf_counter() - started) * 1000
        answer = str(result.get("response") or "")
        rag_trace = result.get("rag_trace") or {}
        status = result.get("status")
        case = {
            "id": sample["id"],
            "question": sample["question"],
            "tags": sample.get("tags", []),
            "run_id": reservation.run_id,
            "status": status,
            "answer": answer,
            "rag_trace": rag_trace,
            "latency_ms": round(elapsed_ms, 1),
        }
        if status != "completed":
            generation_failed += 1
            case["failure_stage"] = "hitl" if status == "waiting_hitl" else "generation"
            cases.append(case)
            continue
        if not answer.strip():
            empty_answers += 1

        if judge is not None:
            try:
                judged = judge_answer(judge, sample, answer, rag_trace)
                case["judge"] = judged
                for metric in METRICS:
                    scores[metric].append(judged[metric])
                scores["unsupported_claims"].append(judged["unsupported_claims"])
            except Exception as exc:
                judge_failed += 1
                case["judge_error"] = str(exc)
                case["failure_stage"] = "judge"
        cases.append(case)
        print(f"  [{index}/{len(rows)}] {sample['id']} -> {status}")

    total = len(rows)
    judged_count = len(scores["correctness"])
    summary = {
        "sample_count": total,
        "completed": sum(case.get("status") == "completed" for case in cases),
        "generation_failure_rate": generation_failed / total,
        "judge_failure_rate": judge_failed / total,
        "empty_answer_rate": empty_answers / total,
        "judged_count": judged_count,
        "judge_metrics": {
            metric: (sum(scores[metric]) / len(scores[metric]) if scores[metric] else None)
            for metric in (*METRICS, "unsupported_claims")
        },
    }
    report = {
        "schema_version": 1,
        "kind": "rag_answer_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_run.isoformat(),
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "generation_model": os.getenv("LLM_MODEL"),
        "judge_model": judge_model_name,
        "summary": summary,
        "cases": cases,
    }

    gates = None
    if args.update_baseline:
        if args.limit or args.no_judge:
            raise ValueError("--update-baseline 不能与 --limit 或 --no-judge 一起使用")
        if judge_failed or generation_failed:
            raise RuntimeError("存在 generation/judge 失败，拒绝更新 baseline")
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Baseline 已更新: {args.baseline}")
    elif args.baseline.exists() and not args.no_gate and not args.no_judge:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        gates = compare_baseline(summary, baseline, args.max_score_regression)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"answer-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    output.write_text(json.dumps({**report, "gates": gates}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已保存: {output}")
    if gates and any(not gate["passed"] for gate in gates):
        print("Baseline gate 失败。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
