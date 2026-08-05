from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from backend.rag.chat_graph import NO_KNOWLEDGE, build_chat_graph
from backend.rag.hitl_detect import HitlDecision


def _initial(user_text="报销标准"):
    return {
        "user_text": user_text,
        "original_question": user_text,
        "question": user_text,
        "answers": [],
        "persistent_note": "",
        "history": [],
        "mode": "",
        "phase": "",
        "docs": [],
        "context": "",
        "rag_trace": {},
        "weather_result": "",
        "response": "",
    }


def _fake_retrieve(state):
    return {
        "docs": [{"filename": "销售.md", "text": "销售提成 10%", "page_number": 1}],
        "context": "销售提成 10%",
        "rag_trace": {"retrieval_status": "ok"},
    }


def _fake_retrieve_with_step(state):
    from langgraph.config import get_stream_writer

    get_stream_writer()({"type": "rag_step", "step": {"icon": "🔍", "label": "检索"}})
    return {
        "docs": [{"filename": "销售.md", "text": "t", "page_number": 1}],
        "context": "t",
        "rag_trace": {"retrieval_status": "ok"},
    }


def _fake_generate(state):
    return {"response": f"回答:{state['question']}"}


def _always_no_hitl(question, docs, router_model=None):
    return HitlDecision(needs_hitl=False)


def _hitl_once(decision):
    calls = {"n": 0}

    def detect(question, docs, router_model=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            return decision
        return HitlDecision(needs_hitl=False)

    return detect


def _build(retrieve_fn=_fake_retrieve, detect_fn=_always_no_hitl, generate_fn=_fake_generate, weather_fn=None):
    return build_chat_graph(
        retrieve_fn=retrieve_fn,
        detect_fn=detect_fn,
        generate_fn=generate_fn,
        weather_fn=weather_fn,
        checkpointer=InMemorySaver(),
    )


def test_plain_question_reaches_generate():
    graph = _build()
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke(_initial("报销标准是什么"), config=cfg)
    assert graph.get_state(cfg).values["response"] == "回答:报销标准是什么"


def test_hitl_interrupt_blocks_and_resumes():
    decision = HitlDecision(
        needs_hitl=True, route="clarify", prompt="请问指的是哪个部门的报销标准？",
        options=[], retrieval_status="needs_clarification",
    )
    graph = _build(detect_fn=_hitl_once(decision))
    cfg = {"configurable": {"thread_id": "t2"}}
    graph.invoke(_initial(), config=cfg)
    snap = graph.get_state(cfg)
    # 图被 interrupt 暂停，未产出答案
    assert snap.tasks and snap.tasks[0].interrupts
    assert snap.tasks[0].interrupts[0].value["route"] == "clarify"
    assert "报销标准" in snap.tasks[0].interrupts[0].value["prompt"]
    assert not snap.values.get("response")
    # resume：补充后 answer 追加进 question，并重新检索→生成
    graph.invoke(Command(resume="技术部的"), config=cfg)
    snap = graph.get_state(cfg)
    assert "技术部的" in snap.values["response"]


def test_nested_hitl_then_resolves():
    decision = HitlDecision(
        needs_hitl=True, route="scope_select", prompt="选一个方向",
        options=["销售", "财务"], retrieval_status="needs_scope_selection",
    )
    calls = {"n": 0}

    def detect(question, docs, router_model=None):
        calls["n"] += 1
        if calls["n"] <= 4:
            return decision
        return HitlDecision(needs_hitl=False)

    graph = _build(detect_fn=detect)
    cfg = {"configurable": {"thread_id": "t3"}}
    graph.invoke(_initial(), config=cfg)
    graph.invoke(Command(resume="销售"), config=cfg)
    # 第二轮仍命中 → 再次 interrupt
    snap = graph.get_state(cfg)
    assert snap.tasks and snap.tasks[0].interrupts
    graph.invoke(Command(resume="销售提成怎么算"), config=cfg)
    snap = graph.get_state(cfg)
    assert not (snap.tasks and snap.tasks[0].interrupts)
    assert "销售" in snap.values["response"]


def test_no_knowledge_when_empty_docs():
    # 用真实 generate（其内置「mode=retrieve 且 docs 为空 → NO_KNOWLEDGE」的提前返回，
    # 不触发 LLM 调用）；generate_fn 传 None 表示用真实节点。
    graph = _build(retrieve_fn=lambda s: {"docs": [], "rag_trace": {}}, generate_fn=None)
    cfg = {"configurable": {"thread_id": "t4"}}
    graph.invoke(_initial(), config=cfg)
    assert graph.get_state(cfg).values["response"] == NO_KNOWLEDGE


def test_weather_route_skips_retrieve():
    graph = _build(
        retrieve_fn=lambda s: (_ for _ in ()).throw(AssertionError("retrieve 不应被调用")),
        weather_fn=lambda s: {"weather_result": "晴", "phase": "generate"},
    )
    cfg = {"configurable": {"thread_id": "t5"}}
    graph.invoke(_initial("武汉今天天气"), config=cfg)
    assert graph.get_state(cfg).values["response"] == "回答:武汉今天天气"


def test_custom_events_from_sync_stream():
    graph = build_chat_graph(
        retrieve_fn=_fake_retrieve_with_step,
        generate_fn=lambda s: s,
        checkpointer=InMemorySaver(),
    )
    events = [d for _k, d in graph.stream(
        {"user_text": "hi", "original_question": "", "question": "", "answers": [],
         "persistent_note": "", "history": [], "mode": "", "phase": "", "docs": [], "context": "",
         "rag_trace": {}, "weather_result": "", "response": ""},
        config={"configurable": {"thread_id": "t3"}}, stream_mode=["custom"])]
    assert any(ev.get("type") == "rag_step" for ev in events)
