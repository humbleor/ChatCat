"""LangSmith 评估脚本：以 ChatCat 真实完整 Agent 流程为评估对象。

运行前提：
  1. 基础设施已启动：`docker compose up -d`（Postgres / Redis / Milvus）
  2. `.env` 已配置 LangSmith 环境变量：
       LANGSMITH_TRACING=true
       LANGSMITH_API_KEY=lsv2_...
       LANGSMITH_PROJECT=ChatCat
  3. LangSmith 中已建好数据集（默认名 "RAG"），每条 example：
       inputs : {"question": "..."}
       outputs: {"answer": "..."}   # 可选参考答案，没有也能跑（只判是否有回答）

运行：
  uv run python eval/eval_langsmith.py
"""

import importlib
import os
import sys
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from langsmith import evaluate

# 项目根目录加入 sys.path，以便使用 backend 包导入
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

# 同步版完整 Agent 流程
chat_with_agent = importlib.import_module("backend.agent.agent").chat_with_agent

# 1. 数据集名（LangSmith 中已存在）
DATASET_NAME = "RAG"


def _extract_answer(outputs: Optional[dict]) -> str:
    """从被测函数返回值中提取最终回答文本。"""
    if not isinstance(outputs, dict):
        return ""
    for key in ("response", "answer", "output"):
        value = outputs.get(key)
        if value:
            return str(value).strip()
    return ""


def _extract_reference(reference_outputs: Optional[dict]) -> str:
    """从数据集 example.outputs 中提取参考答案文本。"""
    if not isinstance(reference_outputs, dict):
        return ""
    for key in ("response", "answer", "output", "expected_answer"):
        value = reference_outputs.get(key)
        if value:
            return str(value).strip()
    return ""


# 2. 自定义评估器（评估最终答案，不评估检索块）
#    langsmith 0.7.x 会按参数名自动注入：
#      outputs           -> run.outputs        （被测函数 target_function 的返回值）
#      reference_outputs -> example.outputs    （数据集里的参考答案）
#    必须返回 {"key": ..., "score": ...} dict，不能返回 bool。
def custom_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    answer = _extract_answer(outputs)
    if not answer or "Retrieved Chunks:" in answer:
        return {"key": "final_answer", "score": 0.0, "comment": "no usable answer"}

    reference = _extract_reference(reference_outputs)
    if not reference:
        return {"key": "final_answer", "score": 1.0, "comment": "answer produced (no reference to compare)"}

    # 有参考答案时，用去空白后的字符集合重合率做轻量语义重合检查
    answer_chars = {ch for ch in answer if not ch.isspace()}
    ref_chars = {ch for ch in reference if not ch.isspace()}
    if not answer_chars or not ref_chars:
        return {"key": "final_answer", "score": 0.0, "comment": "empty normalized answer/reference"}

    overlap = len(answer_chars & ref_chars) / max(1, len(ref_chars))
    score = 1.0 if overlap >= 0.2 else 0.0
    return {"key": "final_answer", "score": score, "comment": f"char-overlap={overlap:.2f}"}


# 3. 被测对象：直接调用现有完整 Agent 流程（含检索、HITL、持久化笔记）
def target_function(inputs: dict) -> dict:
    question = inputs["question"]
    # 每条评估样本使用独立会话，避免上下文串扰
    session_id = f"langsmith_eval_{uuid4().hex}"
    result = chat_with_agent(
        user_text=question,
        user_id="langsmith_eval_user",
        session_id=session_id,
    )

    response_text = ""
    rag_trace = {}
    if isinstance(result, dict):
        response_text = str(result.get("response", "") or "")
        rag_trace = result.get("rag_trace", {}) or {}
    else:
        response_text = str(result)

    return {"response": response_text, "rag_trace": rag_trace}


def main() -> None:
    # 更多说明：https://docs.langchain.com/langsmith/evaluation-concepts
    evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[custom_evaluator],
        experiment_prefix="RAG Pipeline Real Evaluation",
        metadata={"pipeline": "chat_with_agent", "dataset": DATASET_NAME},
        max_concurrency=4,
    )


if __name__ == "__main__":
    main()
