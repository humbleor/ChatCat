"""HITL（Human-in-the-loop）检测：混合判定 + trace 归一化（纯函数，无外部连接）。

- needs_clarification：router LLM 判歧义/缺槽位（由调用方注入 router_model）。
- needs_scope_selection：检索结果按 filename 聚类，多簇即触发。
- no_knowledge：docs 为空。
"""
from dataclasses import dataclass
from typing import Optional

MAX_HITL_ROUNDS = 3
HITL_STATUSES = {"needs_clarification", "needs_scope_selection", "no_knowledge"}
HITL_ROUTES = {"clarify", "scope_select"}

_MAX_STORED_CHUNKS = 10
_MIN_CLUSTER_SIZE = 2  # 簇内最少片段数，低于则视为噪声
_NO_KNOWLEDGE_MSG = "知识库中没有找到可靠的相关信息，暂时无法基于知识库回答这个问题。"


def _safe_text(value) -> str:
    return str(value or "").strip()


def normalize_rag_trace(raw: Optional[dict]) -> dict:
    """归一化 rag_trace：兼容旧 trace、补 HITL 字段安全默认值、截断 chunks。"""
    if not isinstance(raw, dict):
        return {}
    trace = dict(raw)
    trace.setdefault("retrieval_status", None)
    trace.setdefault("hitl_prompt", None)
    trace.setdefault("hitl_options", [])
    if "route" not in trace:
        trace["route"] = None
    chunks = trace.get("retrieved_chunks")
    if isinstance(chunks, list):
        trace["retrieved_chunks"] = chunks[:_MAX_STORED_CHUNKS]
    return trace


def is_hitl_trace(trace: Optional[dict]) -> bool:
    if not isinstance(trace, dict):
        return False
    status = trace.get("retrieval_status")
    route = trace.get("route")
    return status in HITL_STATUSES or route in HITL_ROUTES


def hitl_route_from_trace(trace: dict) -> str:
    status = trace.get("retrieval_status")
    route = trace.get("route")
    if status == "needs_scope_selection" or route == "scope_select":
        return "scope_select"
    return "clarify"


def compose_question(original: str, answers: list[str]) -> str:
    """原始问题 + 各轮 HITL 补充 → 检索/作答用的完整问题。"""
    parts = [original]
    for a in answers:
        parts.append(a)
    return "\n".join(p for p in parts if p.strip())


def format_hitl_message(prompt: str, options: Optional[list[str]] = None) -> str:
    clean_prompt = prompt.strip()
    clean_options = [o.strip() for o in (options or []) if o.strip()]
    if not clean_options:
        return clean_prompt
    option_lines = "\n".join(f"- {o}" for o in clean_options)
    return f"{clean_prompt}\n\n可选方向：\n{option_lines}"


@dataclass
class HitlDecision:
    needs_hitl: bool
    route: str = ""
    prompt: str = ""
    options: list[str] = ()
    retrieval_status: str = ""


def _cluster_docs(docs: list[dict]) -> dict[str, int]:
    """按 filename 聚类并统计簇内片段数。"""
    counts: dict[str, int] = {}
    for doc in docs:
        name = _safe_text(doc.get("filename")) or "Unknown"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _detect_scope(docs: list[dict]) -> Optional[HitlDecision]:
    counts = _cluster_docs(docs)
    qualified = [name for name, n in counts.items() if n >= _MIN_CLUSTER_SIZE]
    if len(qualified) < 2:
        return None
    prompt = "我找到了多个可能相关的知识库方向，请选择你想继续查询的方向。"
    return HitlDecision(
        needs_hitl=True,
        route="scope_select",
        prompt=prompt,
        options=qualified,
        retrieval_status="needs_scope_selection",
    )


def detect_hitl(question: str, docs: list[dict], router_model=None) -> HitlDecision:
    """混合判定。router_model 为 None 时仅走启发式（澄清检测跳过）。"""
    if not docs:
        return HitlDecision(
            needs_hitl=False, route="", retrieval_status="no_knowledge"
        )

    scope = _detect_scope(docs)
    if scope is not None:
        return scope

    if router_model is not None:
        prompt = _judge_clarification(question, router_model)
        if prompt:
            return HitlDecision(
                needs_hitl=True,
                route="clarify",
                prompt=prompt,
                options=[],
                retrieval_status="needs_clarification",
            )

    return HitlDecision(needs_hitl=False)


def _judge_clarification(question: str, router_model) -> Optional[str]:
    """router LLM 判 query 是否歧义/缺槽位；是则返回生成的追问，否则 None。"""
    prompt = (
        "判断下面这个知识库查询是否缺乏回答所需的关键信息（歧义或缺槽位）。\n"
        "如果是，请只输出一句对用户的追问；如果信息已足够，只输出 NO。\n"
        f"查询：{question}"
    )
    try:
        res = router_model.invoke([{"role": "user", "content": prompt}])
        text = _safe_text(getattr(res, "content", str(res)))
        if text.upper().startswith("NO") or not text:
            return None
        return text
    except Exception:
        return None
