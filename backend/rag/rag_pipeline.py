import os
from typing import List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
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

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")
BASE_URL = os.getenv("LLM_BASE_URL")
GRADE_MODEL = os.getenv("LLM_GRADE_MODEL")

_grader_model = None
_router_model = None


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
    'Respond with a JSON object exactly: {{"relevant": "yes|no|unsure", "confidence": <float 0-1>, "reason": "<one sentence>"}}. '
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
        description="step_back=抽象化原问题；hyde=生成假设文档；complex=多实体/多主题问题，拆分为子问题分别检索（默认 step_back）",
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
    rag_trace: Optional[dict]


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
        "rerank_model": retrieve_meta.get("rerank_model"),
        "rerank_endpoint": retrieve_meta.get("rerank_endpoint"),
        "rerank_error": retrieve_meta.get("rerank_error"),
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

    route = "generate_answer" if decision_ok else "rewrite_question"
    if route == "generate_answer":
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
            "evidence_answerability": answerability_labels.get(route, route),
            "rewrite_needed": route == "rewrite_question",
            "evidence_confidence": confidence,
            "evidence_reason": reason,
        }
    )
    return {"route": route, "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:
    question = state["question"]
    emit_rag_step("✏️", "正在重写查询...")
    router = _get_router_model()
    strategy = "step_back"
    strategy_reason = ""
    if router:
        prompt = (
            "请根据用户问题选择最合适的查询扩展策略。\n"
            "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
            "- hyde：模糊、概念性、需要解释或定义的问题。\n"
            "- complex：多实体、多主题的对比或列举类问题（如『A和B的优缺点』『A、B、C 三者区别』），需要拆分为子问题分别检索。\n"
            '严格输出 JSON：{"strategy": "step_back|hyde|complex", "reason": "一句话选择理由"}，不要其它文字或 markdown。\n'
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
        decomposed = decompose_question(question)
        sub_questions = decomposed.get("sub_questions") or [question]
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
            "sub_agent_count": len(sub_questions),
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
    rerank_model = None
    rerank_endpoint = None
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
        rerank_model = rerank_model or hyde_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or hyde_meta.get("rerank_endpoint")
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
        rerank_model = rerank_model or step_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or step_meta.get("rerank_endpoint")
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

    if strategy == "complex":
        sub_questions = state.get("sub_questions") or [state["question"]]
        total = len(sub_questions)
        for idx, sub_query in enumerate(sub_questions, 1):
            if not sub_query:
                continue
            retrieved_sub = retrieve_documents(sub_query, top_k=RETRIEVAL_TOP_K)
            results.extend(retrieved_sub.get("docs", []))
            sub_meta = retrieved_sub.get("meta", {})
            emit_rag_step(
                "🧱",
                f"子问题 {idx}/{total} 三级检索",
                (
                    f"L{sub_meta.get('leaf_retrieve_level', 3)} 召回，"
                    f"候选 {sub_meta.get('candidate_k', 0)}，"
                    f"合并替换 {sub_meta.get('auto_merge_replaced_chunks', 0)}"
                ),
            )
            rerank_applied_any = rerank_applied_any or bool(sub_meta.get("rerank_applied"))
            rerank_enabled_any = rerank_enabled_any or bool(sub_meta.get("rerank_enabled"))
            rerank_model = rerank_model or sub_meta.get("rerank_model")
            rerank_endpoint = rerank_endpoint or sub_meta.get("rerank_endpoint")
            if sub_meta.get("rerank_error"):
                rerank_errors.append(f"subq{idx}:{sub_meta.get('rerank_error')}")
            retrieval_mode = retrieval_mode or sub_meta.get("retrieval_mode")
            candidate_k = candidate_k or sub_meta.get("candidate_k")
            leaf_retrieve_level = leaf_retrieve_level or sub_meta.get("leaf_retrieve_level")
            auto_merge_enabled = (
                auto_merge_enabled if auto_merge_enabled is not None else sub_meta.get("auto_merge_enabled")
            )
            auto_merge_applied = auto_merge_applied or bool(sub_meta.get("auto_merge_applied"))
            auto_merge_threshold = auto_merge_threshold or sub_meta.get("auto_merge_threshold")
            auto_merge_replaced_chunks += int(sub_meta.get("auto_merge_replaced_chunks") or 0)
            auto_merge_steps += int(sub_meta.get("auto_merge_steps") or 0)

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
            "rerank_model": rerank_model,
            "rerank_endpoint": rerank_endpoint,
            "rerank_error": "; ".join(rerank_errors) if rerank_errors else None,
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


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded)

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
    graph.add_edge("rewrite_question", "retrieve_expanded")
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
            "rag_trace": None,
        }
    )
