"""统一对话 StateGraph：prepare_context → route → {weather,retrieve→check_hitl} → generate。

HITL 用 LangGraph 原生 interrupt/Command(resume)；节点全同步，RAG 步骤与内容
通过 get_stream_writer() 发 custom 事件，供编排层转 SSE。
"""

import re
from typing import Callable, Optional, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from backend.rag.hitl_detect import (
    HitlDecision,
    compose_question,
    normalize_rag_trace,
)

NO_KNOWLEDGE = "知识库中没有找到可靠的相关信息，暂时无法基于知识库回答这个问题。"


class ChatState(TypedDict):
    user_text: str
    original_question: str
    question: str
    answers: list[str]
    persistent_note: str
    history: list[dict]
    mode: str
    phase: str
    docs: list[dict]
    context: str
    rag_trace: dict
    weather_result: str
    response: str


# ---------- 模型（懒加载，避免 import 期外部依赖） ----------
_chat_model = None


def _get_chat_model():
    global _chat_model
    if _chat_model is None:
        import os

        from langchain.chat_models import init_chat_model

        _chat_model = init_chat_model(
            model=os.getenv("LLM_MODEL"),
            model_provider="openai",
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0.3,
        )
    return _chat_model


# ---------- 节点 ----------
def prepare_context(state: ChatState) -> dict:
    return {
        "original_question": state["user_text"],
        "question": state["user_text"],
        "answers": [],
        "mode": "",
        "phase": "",
        "docs": [],
        "context": "",
        "rag_trace": {},
        "weather_result": "",
        "response": "",
    }


_WEATHER_KEYWORDS = ("天气", "气温", "温度", "下雨", "下雪", "weather", "forecast", "climate")


def _judge_route(question: str) -> str:
    q = question.strip().lower()
    if any(k in q for k in _WEATHER_KEYWORDS):
        return "weather"
    try:
        from backend.rag.rag_pipeline import _get_router_model

        router = _get_router_model()
        if router is not None:
            prompt = (
                "判断以下用户问题是否需要检索知识库文档。\n"
                "若问题涉及特定文档/资料/知识库内容 → 只输出 retrieve；"
                "否则（寒暄、闲聊、观点、天气等）→ 只输出 generate。\n"
                f"用户问题：{question}"
            )
            res = router.invoke([{"role": "user", "content": prompt}])
            text = str(getattr(res, "content", str(res)) or "").strip().lower()
            if "retrieve" in text:
                return "retrieve"
            if "generate" in text:
                return "generate"
    except Exception:
        pass
    return "retrieve"  # 兜底：默认尝试检索


def route(state: ChatState) -> dict:
    mode = _judge_route(state["question"])
    return {"mode": mode, "phase": mode}


def _history_to_messages(history: list[dict]):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    out = []
    for item in history:
        role = (item.get("role") or "").lower()
        content = str(item.get("content") or "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


def _format_docs(docs: list[dict]) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        src = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        lines.append(f"[{i}] {src} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(lines)


def retrieve(state: ChatState) -> dict:
    from backend.rag.rag_pipeline import run_rag_graph

    writer = get_stream_writer()
    step = {"icon": "🔍", "label": "正在检索知识库...", "detail": f"查询: {state['question'][:50]}"}
    writer({"type": "rag_step", "step": step})
    result = run_rag_graph(state["question"])
    docs = result.get("docs", []) if isinstance(result, dict) else []
    trace = normalize_rag_trace(result.get("rag_trace") if isinstance(result, dict) else None)
    writer({"type": "rag_step", "step": {"icon": "✅", "label": f"检索完成，找到 {len(docs)} 个片段"}})
    return {"docs": docs, "context": _format_docs(docs), "rag_trace": trace}


def _default_detect(question: str, docs: list[dict], router_model=None) -> HitlDecision:
    """真实检测：澄清用 router LLM（懒加载），范围选择用聚类启发式。"""
    from backend.rag.hitl_detect import detect_hitl
    from backend.rag.rag_pipeline import _get_router_model

    return detect_hitl(question, docs, router_model=_get_router_model())


def _make_check_hitl(detect_fn: Callable):
    """构造 check_hitl 节点。detect_fn 注入（测试传 fake，生产传 _default_detect），
    interrupt 逻辑唯一一份，避免 DRY 破坏。"""

    def check_hitl(state: ChatState) -> dict:
        from backend.rag.hitl_detect import MAX_HITL_ROUNDS

        if len(state["answers"]) >= MAX_HITL_ROUNDS:
            return {"phase": "generate"}

        decision: HitlDecision = detect_fn(state["question"], state["docs"])
        if not decision.needs_hitl:
            return {"phase": "generate"}

        hitl_value = {"route": decision.route, "prompt": decision.prompt, "options": list(decision.options)}
        answer = interrupt(hitl_value)
        answers = [*state["answers"], answer]
        trace = dict(state["rag_trace"] or {})
        trace.update(
            {
                "retrieval_status": decision.retrieval_status,
                "route": decision.route,
                "hitl_prompt": decision.prompt,
                "hitl_options": list(decision.options),
            }
        )
        return {
            "answers": answers,
            "question": compose_question(state["original_question"], answers),
            "phase": "retrieve",
            "rag_trace": trace,
        }

    return check_hitl


# ---------- 天气城市名提取 ----------
# 保守快速路径：只接受「干净城市名 + 天气/时间词」句式；含动词/副词/时间词的匹配一律拒绝，
# 交给 LLM 兜底，避免把「知道武汉」「武汉下周三」这类脏片段直接传给天气工具。
_LEADING_NOISE_RE = re.compile(
    r"^(?:请问|麻烦|帮忙|帮我|帮我查|查一?下|看看|看下|一下|今天|明天|后天|昨天|现在|想知道|我想)+"
)
_FAST_LOC_RE = re.compile(
    r"^([一-鿿]{2,6}?)(?:的)?(?:今天|明天|后天|昨天|现在|天气|气温|温度|下雨|下雪|风力|风向|湿度|预报)"
)
_DIRTY_LOC_RE = re.compile(
    r"查|知|看|会|下|周|请|想|帮|问|麻烦|帮忙|预报|今|明|后"
    r"|天气|气温|温度|下雨|下雪|风力|风向|湿度|怎么样|多少|什么"
)


def _extract_location_fast(question: str) -> Optional[str]:
    """快速路径：只处理「城市 + 天气/时间词」这种直白句式，复杂句式交给 LLM。"""
    q = _LEADING_NOISE_RE.sub("", question.strip())
    m = _FAST_LOC_RE.match(q)
    if not m:
        return None
    loc = m.group(1)
    # 拒绝包含动词/副词/时间词/天气词或并列地名的「脏」城市名
    if 2 <= len(loc) <= 5 and not _DIRTY_LOC_RE.search(loc) and not any(c in loc for c in "和与及、，,"):
        return loc
    return None


def _extract_location_llm(question: str) -> Optional[str]:
    """LLM 兜底：懒加载 router 模型，一次性抽取城市名（只回城市名或 NONE）。"""
    try:
        from backend.rag.rag_pipeline import _get_router_model

        router_model = _get_router_model()
        if router_model is None:
            return None
        prompt = (
            "从以下天气查询中提取城市名。只输出城市名本身（例如：武汉），不要任何解释。"
            "如果查询中没有明确的城市名，只输出 NONE。\n"
            f"查询：{question}"
        )
        res = router_model.invoke([{"role": "user", "content": prompt}])
        text = str(getattr(res, "content", "") or "").strip()
        if not text:
            return None
        first_line = text.splitlines()[0].strip()
        if first_line.upper() == "NONE":
            return None
        # 剥离可能带出的天气/时间尾巴，只留城市名
        cleaned = re.sub(
            r"(?:的)?(?:今天|明天|后天|昨天|现在|天气|气温|温度|下雨|下雪|风力|风向|湿度|预报|怎么样|多少)+$",
            "",
            first_line,
        )
        cleaned = cleaned.strip("。.,，:： \t")
        return cleaned if 1 <= len(cleaned) <= 8 else None
    except Exception:
        return None


def _extract_location(question: str) -> str:
    """从问句中提取城市名。返回城市名；无法识别返回空字符串。"""
    loc = _extract_location_fast(question)
    if loc:
        return loc
    loc = _extract_location_llm(question)
    return loc or ""


def weather_node(state: ChatState) -> dict:
    from backend.agent.tools import get_current_weather

    location = _extract_location(state["question"])
    if not location:
        return {"weather_result": "无法识别查询中的城市", "phase": "generate"}
    result = get_current_weather.func(location, extensions="base")
    return {"weather_result": result, "phase": "generate"}


def generate(state: ChatState) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    writer = get_stream_writer()
    mode = state["mode"]

    if mode == "retrieve":
        if not state["docs"]:
            writer({"type": "content", "content": NO_KNOWLEDGE})
            return {"response": NO_KNOWLEDGE}
        system = SystemMessage(
            content=(
                "You are a helpful knowledge-base assistant. "
                "Answer the user's question using ONLY the retrieved chunks. "
                "Cite source chunks inline with [1], [2], etc. "
                "If the chunks are insufficient, say so honestly. "
                "Do not mention internal HITL or RAG implementation details."
            )
        )
        human = HumanMessage(content=f"问题：\n{state['question']}\n\n检索片段：\n{state['context']}")
        messages = [system, *_history_to_messages(state["history"]), human]
    elif mode == "weather":
        human = HumanMessage(content=f"问题：\n{state['question']}\n\n天气数据：\n{state['weather_result']}")
        messages = [
            HumanMessage(content="你是天气助手，基于天气数据简洁回答。"),
            *_history_to_messages(state["history"]),
            human,
        ]
    else:
        messages = []
        if state["persistent_note"]:
            messages.append(
                SystemMessage(
                    content=f"【对话持久化笔记（你的工作记忆）】\n{state['persistent_note']}\n请参考以上笔记保持对话连贯性。"
                )
            )
        messages.extend(_history_to_messages(state["history"]))
        messages.append(HumanMessage(content=state["question"]))

    model = _get_chat_model()
    full = ""
    try:
        for chunk in model.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if isinstance(content, list):
                text = "".join(
                    (
                        b.get("text", "")
                        if isinstance(b, dict) and b.get("type") == "text"
                        else (b if isinstance(b, str) else "")
                    )
                    for b in content
                )
            else:
                text = str(content)
            if text:
                full += text
                writer({"type": "content", "content": text})
    except Exception as e:
        writer({"type": "content", "content": f"\n[Error: {e}]"})
        full += f"\n[Error: {e}]"
    return {"response": full}


# ---------- 图装配（依赖注入，测试可替换节点） ----------
def build_chat_graph(
    *,
    retrieve_fn: Optional[Callable] = None,
    detect_fn: Optional[Callable] = None,
    generate_fn: Optional[Callable] = None,
    weather_fn: Optional[Callable] = None,
    checkpointer=None,
):
    """retrieve_fn/detect_fn/generate_fn/weather_fn 用于测试注入；默认用真实节点。

    detect_fn 签名：(question, docs, router_model=None) -> HitlDecision。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    _retrieve = retrieve_fn or retrieve
    _generate = generate_fn or generate
    _weather = weather_fn or weather_node
    _detect = detect_fn or _default_detect
    _check_hitl = _make_check_hitl(_detect)

    g = StateGraph(ChatState)
    g.add_node("prepare_context", prepare_context)
    g.add_node("route", route)
    g.add_node("retrieve", _retrieve)
    g.add_node("check_hitl", _check_hitl)
    g.add_node("weather", _weather)
    g.add_node("generate", _generate)

    g.add_edge(START, "prepare_context")
    g.add_edge("prepare_context", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["phase"],
        {"retrieve": "retrieve", "weather": "weather", "generate": "generate"},
    )
    g.add_edge("retrieve", "check_hitl")
    g.add_conditional_edges(
        "check_hitl",
        lambda s: s["phase"],
        {"retrieve": "retrieve", "generate": "generate"},
    )
    g.add_edge("weather", "generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
