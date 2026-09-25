import os
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from backend.agent.tools import emit_rag_step
from backend.rag.rag_utils import (
    RETRIEVAL_TOP_K,
    decompose_question,
    generate_hypothetical_document,
    retrieve_documents,
    step_back_expand,
    strip_think,
)

API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")
BASE_URL = os.getenv("LLM_BASE_URL")
GRADE_MODEL = os.getenv("LLM_GRADE_MODEL")

_grader_model = None
_router_model = None
MAX_PARALLEL_SUBTASKS = 3


def _merge_sub_results(left: dict, right: dict) -> dict:
    """并行分支按子任务 ID 合并；相同 ID 的新结果覆盖旧值。"""
    return {**left, **right}


def _get_grader_model():
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _grader_model


def _get_router_model():
    global _router_model
    if not API_KEY or not MODEL:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _router_model


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n"
    'Respond with a JSON object exactly: {{"relevant": "yes|no|unsure", '
    '"confidence": <float 0-1>, "reason": "<one sentence>"}}. '
    "Do not include any other text or markdown."
)


class GradeResult(BaseModel):
    """文档相关性判定（LLM 结构化输出，取代旧 _parse_grade 子串匹配）。"""

    relevant: Literal["yes", "no", "unsure"] = Field(
        default="unsure",
        description="yes=明确相关；no=明确不相关；unsure=拿不准（默认值）",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="置信度 0-1；>=0.6 视为高置信度",
    )
    reason: str = Field(default="", description="判断理由（一句话），写入 rag_trace 方便 debug")


class StrategyResult(BaseModel):
    """查询扩展策略选择（LLM 结构化输出，取代旧 _parse_strategy 子串匹配）。"""

    strategy: Literal["step_back", "hyde", "complex"] = Field(
        default="step_back",
        description=(
            "step_back=抽象化原问题；hyde=生成假设文档；"
            "complex=多实体/多主题问题，拆分为子问题分别检索（默认 step_back）"
        ),
    )
    reason: str = Field(default="", description="选择理由（一句话）")


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    sub_questions: Optional[List[str]]
    sub_results: Annotated[dict[str, dict], _merge_sub_results]
    rag_trace: Optional[dict]


class SubtaskState(TypedDict):
    subtask_id: str
    query: str


def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)


def retrieve_initial(state: RAGState) -> RAGState:
    query = state["question"]
    emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")
    retrieved = retrieve_documents(query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)
    emit_rag_step(
        "🧱",
        "三级分块检索",
        (f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，候选 {retrieve_meta.get('candidate_k', 0)}"),
    )
    emit_rag_step(
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    emit_rag_step(
        "✅", f"检索完成，找到 {len(results)} 个片段", f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}"
    )
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "expanded_query": query,
        "retrieved_chunks": results,
        "initial_retrieved_chunks": results,
        "retrieval_stage": "initial",
        "rerank_enabled": retrieve_meta.get("rerank_enabled"),
        "rerank_applied": retrieve_meta.get("rerank_applied"),
        "rerank_provider": retrieve_meta.get("rerank_provider"),
        "rerank_model": retrieve_meta.get("rerank_model"),
        "rerank_endpoint": retrieve_meta.get("rerank_endpoint"),
        "rerank_error": retrieve_meta.get("rerank_error"),
        "rerank_latency_ms": retrieve_meta.get("rerank_latency_ms"),
        "retrieval_mode": retrieve_meta.get("retrieval_mode"),
        "candidate_k": retrieve_meta.get("candidate_k"),
        "leaf_retrieve_level": retrieve_meta.get("leaf_retrieve_level"),
        "auto_merge_enabled": retrieve_meta.get("auto_merge_enabled"),
        "auto_merge_applied": retrieve_meta.get("auto_merge_applied"),
        "auto_merge_threshold": retrieve_meta.get("auto_merge_threshold"),
        "auto_merge_replaced_chunks": retrieve_meta.get("auto_merge_replaced_chunks"),
        "auto_merge_steps": retrieve_meta.get("auto_merge_steps"),
    }
    return {
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    grader = _get_grader_model()
    emit_rag_step("📊", "正在评估文档相关性...")
    question = state["question"]
    context = state.get("context", "")

    # 默认评分（LLM 失败时按"不能确定 → 重写"保守策略，与原逻辑一致）
    decision_ok = False
    score_label = "unknown"
    confidence = 0.0
    reason = ""

    try:
        if not grader:
            raise RuntimeError("grader 模型未配置")
        prompt = GRADE_PROMPT.format(question=question, context=context)
        # 推理模型会先输出 <think>...</think> 推理块，剥掉再喂给 Pydantic 校验。
        raw = grader.invoke([{"role": "user", "content": prompt}]).content or ""
        cleaned = strip_think(raw)
        result: GradeResult = GradeResult.model_validate_json(cleaned)
        # yes + 高 confidence 才放行；unsure + 极高 confidence 也可放行
        decision_ok = (result.relevant == "yes" and result.confidence >= 0.6) or (
            result.relevant == "unsure" and result.confidence >= 0.85
        )
        score_label = result.relevant
        confidence = result.confidence
        reason = result.reason
    except Exception as e:
        # 打分失败时降级为"需要重写查询"，与原逻辑一致
        emit_rag_step("⚠️", "评估异常，降级到重写", f"err: {str(e)[:80]}")

    # 相关性评分只说明检索到了相关内容，不能保证比较题的每个实体都有证据。
    # 明确的比较题先尝试拆分；确有多个独立子问题时交给并行检索。
    planned_sub_questions: List[str] = []
    if decision_ok and any(marker in question for marker in ("区别", "比较", "对比", "异同")):
        decomposed = decompose_question(question)
        planned_sub_questions = list(dict.fromkeys(q.strip() for q in decomposed.get("sub_questions", []) if q.strip()))
        if len(planned_sub_questions) < 2:
            planned_sub_questions = []

    route = "rewrite_question" if planned_sub_questions or not decision_ok else "generate_answer"
    if planned_sub_questions:
        emit_rag_step("🔀", "比较题需要多路证据", f"规划了 {len(planned_sub_questions)} 个子问题")
    elif route == "generate_answer":
        emit_rag_step("✅", "文档相关性评估通过", f"评分: {score_label} (conf={confidence:.2f})")
    else:
        emit_rag_step("⚠️", "文档相关性不足，将重写查询", f"评分: {score_label} (conf={confidence:.2f})")

    # grade_* → evidence_* 重命名：字段名对齐前端 RagTraceFields，语义用中文标签固化（前端直显）。
    relevance_labels = {"yes": "相关", "no": "不相关", "unsure": "拿不准"}
    answerability_labels = {"generate_answer": "可回答", "rewrite_question": "需要改写"}
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(
        {
            "evidence_relevance": relevance_labels.get(score_label, score_label),
            "evidence_answerability": (
                "需要多路证据" if planned_sub_questions else answerability_labels.get(route, route)
            ),
            "rewrite_needed": route == "rewrite_question",
            "evidence_confidence": confidence,
            "evidence_reason": reason,
        }
    )
    return {"route": route, "sub_questions": planned_sub_questions, "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:
    question = state["question"]
    emit_rag_step("✏️", "正在重写查询...")
    router = _get_router_model()
    planned_sub_questions = state.get("sub_questions") or []
    strategy = "complex" if len(planned_sub_questions) > 1 else "step_back"
    strategy_reason = "比较题已拆分为多个独立子问题" if len(planned_sub_questions) > 1 else ""
    if router and not planned_sub_questions:
        prompt = (
            "请根据用户问题选择最合适的查询扩展策略。\n"
            "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
            "- hyde：模糊、概念性、需要解释或定义的问题。\n"
            "- complex：多实体、多主题的对比或列举类问题"
            "（如『A和B的优缺点』『A、B、C 三者区别』），需要拆分为子问题分别检索。\n"
            '严格输出 JSON：{"strategy": "step_back|hyde|complex", '
            '"reason": "一句话选择理由"}，不要其它文字或 markdown。\n'
            f"用户问题：{question}"
        )
        try:
            # 推理模型会先输出 <think>...</think> 推理块，剥掉再喂给 Pydantic 校验，
            # 否则 message.content 开头是 '<' 不是 '{'，ValidationError 抛出。
            raw = router.invoke([{"role": "user", "content": prompt}]).content or ""
            cleaned = strip_think(raw)
            result: StrategyResult = StrategyResult.model_validate_json(cleaned)
            strategy = result.strategy
            strategy_reason = result.reason
        except Exception as e:
            strategy = "step_back"
            strategy_reason = f"Router 异常({type(e).__name__}): {str(e)[:120]} 或默认 step_back"

    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""
    sub_questions: List[str] = []

    if strategy == "complex":
        emit_rag_step("🔪", f"使用策略: {strategy}", "分解为子问题")
        decomposed = {"sub_questions": planned_sub_questions} if planned_sub_questions else decompose_question(question)
        sub_questions = [q.strip() for q in (decomposed.get("sub_questions") or [question]) if q.strip()]
        sub_questions = list(dict.fromkeys(sub_questions)) or [question]
        if len(sub_questions) > MAX_PARALLEL_SUBTASKS:
            sub_questions = [
                *sub_questions[: MAX_PARALLEL_SUBTASKS - 1],
                "；".join(sub_questions[MAX_PARALLEL_SUBTASKS - 1 :]),
            ]
        if len(sub_questions) > 1:
            preview = " → ".join(sub_questions[:3])
            if len(sub_questions) > 3:
                preview += " ..."
            emit_rag_step("✅", f"已分解为 {len(sub_questions)} 个子问题", preview)
        else:
            emit_rag_step("✅", "无需分解，保持原问题", sub_questions[0])

    if strategy == "step_back":
        emit_rag_step("🧠", f"使用策略: {strategy}", "生成退步问题")
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)

    if strategy == "hyde":
        emit_rag_step("📝", "HyDE 假设性文档生成中...")
        hypothetical_doc = generate_hypothetical_document(question)

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(
        {
            "rewrite_strategy": strategy,
            "rewrite_query": expanded_query,
            "rewrite_reason": strategy_reason,
            "sub_questions": sub_questions,
            "sub_agent_count": len(sub_questions) if strategy == "complex" else 0,
        }
    )

    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "sub_questions": sub_questions,
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    strategy = state.get("expansion_type") or "step_back"
    emit_rag_step("🔄", "使用扩展查询重新检索...", f"策略: {strategy}")
    results: List[dict] = []
    rerank_applied_any = False
    rerank_enabled_any = False
    rerank_provider = None
    rerank_model = None
    rerank_endpoint = None
    rerank_latency_ms = 0.0
    rerank_errors = []
    retrieval_mode = None
    candidate_k = None
    leaf_retrieve_level = None
    auto_merge_enabled = None
    auto_merge_applied = False
    auto_merge_threshold = None
    auto_merge_replaced_chunks = 0
    auto_merge_steps = 0

    if strategy == "hyde":
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        retrieved_hyde = retrieve_documents(hypothetical_doc, top_k=RETRIEVAL_TOP_K)
        results.extend(retrieved_hyde.get("docs", []))
        hyde_meta = retrieved_hyde.get("meta", {})
        emit_rag_step(
            "🧱",
            "HyDE 三级检索",
            (
                f"L{hyde_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {hyde_meta.get('candidate_k', 0)}，"
                f"合并替换 {hyde_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_applied_any = rerank_applied_any or bool(hyde_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(hyde_meta.get("rerank_enabled"))
        rerank_provider = rerank_provider or hyde_meta.get("rerank_provider")
        rerank_model = rerank_model or hyde_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or hyde_meta.get("rerank_endpoint")
        rerank_latency_ms += float(hyde_meta.get("rerank_latency_ms") or 0.0)
        if hyde_meta.get("rerank_error"):
            rerank_errors.append(f"hyde:{hyde_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or hyde_meta.get("retrieval_mode")
        candidate_k = candidate_k or hyde_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or hyde_meta.get("leaf_retrieve_level")
        auto_merge_enabled = (
            auto_merge_enabled if auto_merge_enabled is not None else hyde_meta.get("auto_merge_enabled")
        )
        auto_merge_applied = auto_merge_applied or bool(hyde_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or hyde_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(hyde_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(hyde_meta.get("auto_merge_steps") or 0)

    if strategy == "step_back":
        expanded_query = state.get("expanded_query") or state["question"]
        retrieved_stepback = retrieve_documents(expanded_query, top_k=RETRIEVAL_TOP_K)
        results.extend(retrieved_stepback.get("docs", []))
        step_meta = retrieved_stepback.get("meta", {})
        emit_rag_step(
            "🧱",
            "Step-back 三级检索",
            (
                f"L{step_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {step_meta.get('candidate_k', 0)}，"
                f"合并替换 {step_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_applied_any = rerank_applied_any or bool(step_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(step_meta.get("rerank_enabled"))
        rerank_provider = rerank_provider or step_meta.get("rerank_provider")
        rerank_model = rerank_model or step_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or step_meta.get("rerank_endpoint")
        rerank_latency_ms += float(step_meta.get("rerank_latency_ms") or 0.0)
        if step_meta.get("rerank_error"):
            rerank_errors.append(f"step_back:{step_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or step_meta.get("retrieval_mode")
        candidate_k = candidate_k or step_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or step_meta.get("leaf_retrieve_level")
        auto_merge_enabled = (
            auto_merge_enabled if auto_merge_enabled is not None else step_meta.get("auto_merge_enabled")
        )
        auto_merge_applied = auto_merge_applied or bool(step_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or step_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(step_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(step_meta.get("auto_merge_steps") or 0)

    deduped = []
    seen = set()
    for item in results:
        key = (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # 扩展阶段可能合并了多路召回（如 hyde + step_back），
    # 这里统一重排展示名次，避免出现 1,2,3,4,5,4,5 这类重复名次。
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    context = _format_docs(deduped)
    emit_rag_step("✅", f"扩展检索完成，共 {len(deduped)} 个片段")
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(
        {
            "expanded_query": state.get("expanded_query") or state["question"],
            "step_back_question": state.get("step_back_question", ""),
            "step_back_answer": state.get("step_back_answer", ""),
            "hypothetical_doc": state.get("hypothetical_doc", ""),
            "expansion_type": strategy,
            "retrieved_chunks": deduped,
            "expanded_retrieved_chunks": deduped,
            "retrieval_stage": "expanded",
            "rerank_enabled": rerank_enabled_any,
            "rerank_applied": rerank_applied_any,
            "rerank_provider": rerank_provider,
            "rerank_model": rerank_model,
            "rerank_endpoint": rerank_endpoint,
            "rerank_error": "; ".join(rerank_errors) if rerank_errors else None,
            "rerank_latency_ms": round(rerank_latency_ms, 1),
            "retrieval_mode": retrieval_mode,
            "candidate_k": candidate_k,
            "leaf_retrieve_level": leaf_retrieve_level,
            "auto_merge_enabled": auto_merge_enabled,
            "auto_merge_applied": auto_merge_applied,
            "auto_merge_threshold": auto_merge_threshold,
            "auto_merge_replaced_chunks": auto_merge_replaced_chunks,
            "auto_merge_steps": auto_merge_steps,
            "synthesis_merged_count": len(deduped),
        }
    )
    return {"docs": deduped, "context": context, "rag_trace": rag_trace}


def _dispatch_expansion(state: RAGState):
    if state.get("expansion_type") != "complex":
        return "retrieve_expanded"
    questions = state.get("sub_questions") or [state["question"]]
    return [
        Send("retrieve_subquestion", {"subtask_id": str(index), "query": query})
        for index, query in enumerate(questions[:MAX_PARALLEL_SUBTASKS])
    ]


def retrieve_subquestion(state: SubtaskState) -> dict:
    """隔离子任务：只写以 ID 为键的结果，不改共享文档或 trace。"""
    subtask_id = state["subtask_id"]
    query = state["query"]
    try:
        retrieved = retrieve_documents(query, top_k=RETRIEVAL_TOP_K)
        docs = retrieved.get("docs") or []
        meta = retrieved.get("meta") or {}
        status = "failed" if meta.get("retrieval_mode") == "failed" else ("ok" if docs else "empty")
        error = str(meta.get("rerank_error") or "") if status == "failed" else ""
    except Exception as exc:
        docs, meta, status = [], {}, "failed"
        error = f"{type(exc).__name__}: {exc}"[:200]
    status_label = {"ok": "检索完成", "empty": "未找到证据", "failed": "检索失败"}[status]
    emit_rag_step("🧱" if status == "ok" else "⚠️", f"子问题 {int(subtask_id) + 1} {status_label}", query[:80])
    return {
        "sub_results": {
            subtask_id: {
                "query": query,
                "docs": docs,
                "meta": meta,
                "status": status,
                "error": error,
            }
        }
    }


def synthesize_subquestions(state: RAGState) -> dict:
    """唯一的证据汇总节点；按规划顺序整理结果，避免并发完成顺序影响引用。"""
    questions = state.get("sub_questions") or [state["question"]]
    sub_results = state.get("sub_results") or {}
    ordered = [sub_results.get(str(index), {}) for index in range(len(questions))]
    seen = set()
    merged = []
    # 轮流取各子问题证据，避免第一路占满上下文。
    max_docs = max((len(result.get("docs") or []) for result in ordered), default=0)
    for rank in range(max_docs):
        for result in ordered:
            docs = result.get("docs") or []
            if rank >= len(docs):
                continue
            item = docs[rank]
            key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(item))
    for index, item in enumerate(merged, 1):
        item["rrf_rank"] = index

    missing = [question for question, result in zip(questions, ordered) if not result.get("docs")]
    failures = [question for question, result in zip(questions, ordered) if result.get("status") == "failed"]
    metas = [result.get("meta") or {} for result in ordered]
    traces = [
        {
            "query": question,
            "status": result.get("status", "failed"),
            "error": result.get("error") or None,
            "retrieval_stage": "expanded",
            "retrieval_mode": (result.get("meta") or {}).get("retrieval_mode"),
            "retrieved_chunks": (result.get("docs") or [])[:3],
        }
        for question, result in zip(questions, ordered)
    ]
    trace = dict(state.get("rag_trace") or {})
    trace.update(
        {
            "expanded_query": state["question"],
            "expansion_type": "complex",
            "retrieval_stage": "expanded",
            "retrieved_chunks": merged,
            "expanded_retrieved_chunks": merged,
            "sub_agent_count": len(questions),
            "sub_traces": traces,
            "missing_sub_questions": missing,
            "failed_sub_questions": failures,
            "synthesis_merged_count": len(merged),
            "retrieval_empty": not merged,
            "retrieval_mode": "parallel_subquestions",
            "rerank_enabled": any(bool(meta.get("rerank_enabled")) for meta in metas),
            "rerank_applied": any(bool(meta.get("rerank_applied")) for meta in metas),
            "rerank_provider": next((meta["rerank_provider"] for meta in metas if meta.get("rerank_provider")), None),
            "rerank_model": next((meta["rerank_model"] for meta in metas if meta.get("rerank_model")), None),
            "rerank_endpoint": next((meta["rerank_endpoint"] for meta in metas if meta.get("rerank_endpoint")), None),
            "rerank_error": "; ".join(
                f"subq{index + 1}:{meta['rerank_error']}"
                for index, meta in enumerate(metas)
                if meta.get("rerank_error")
            )
            or None,
            "rerank_latency_ms": round(sum(float(meta.get("rerank_latency_ms") or 0) for meta in metas), 1),
            "candidate_k": max((int(meta.get("candidate_k") or 0) for meta in metas), default=0),
            "leaf_retrieve_level": next(
                (meta["leaf_retrieve_level"] for meta in metas if meta.get("leaf_retrieve_level") is not None),
                None,
            ),
            "auto_merge_enabled": any(bool(meta.get("auto_merge_enabled")) for meta in metas),
            "auto_merge_applied": any(bool(meta.get("auto_merge_applied")) for meta in metas),
            "auto_merge_threshold": next(
                (meta["auto_merge_threshold"] for meta in metas if meta.get("auto_merge_threshold") is not None),
                None,
            ),
            "auto_merge_replaced_chunks": sum(int(meta.get("auto_merge_replaced_chunks") or 0) for meta in metas),
            "auto_merge_steps": sum(int(meta.get("auto_merge_steps") or 0) for meta in metas),
        }
    )
    emit_rag_step(
        "✅" if not missing else "⚠️", f"子问题证据汇总：{len(merged)} 个片段", f"{len(missing)} 个子问题缺少证据"
    )
    return {"docs": merged, "context": _format_docs(merged), "rag_trace": trace}


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded)
    graph.add_node("retrieve_subquestion", retrieve_subquestion)
    graph.add_node("synthesize_subquestions", synthesize_subquestions)

    graph.set_entry_point("retrieve_initial")
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_conditional_edges("rewrite_question", _dispatch_expansion)
    graph.add_edge("retrieve_subquestion", "synthesize_subquestions")
    graph.add_edge("synthesize_subquestions", END)
    graph.add_edge("retrieve_expanded", END)
    return graph.compile()


rag_graph = build_rag_graph()


def run_rag_graph(question: str) -> dict:
    return rag_graph.invoke(
        {
            "question": question,
            "query": question,
            "context": "",
            "docs": [],
            "route": None,
            "expansion_type": None,
            "expanded_query": None,
            "step_back_question": None,
            "step_back_answer": None,
            "hypothetical_doc": None,
            "sub_questions": None,
            "sub_results": {},
            "rag_trace": None,
        }
    )
