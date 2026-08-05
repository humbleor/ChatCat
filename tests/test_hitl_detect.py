from backend.rag.hitl_detect import (
    compose_question,
    detect_hitl,
    format_hitl_message,
    hitl_route_from_trace,
    is_hitl_trace,
    normalize_rag_trace,
)


def test_normalize_none():
    assert normalize_rag_trace(None) == {}


def test_normalize_keeps_legacy_route_not_hitl():
    trace = normalize_rag_trace({"route": "generate_answer", "grade_score": "yes"})
    assert trace["route"] == "generate_answer"
    assert not is_hitl_trace(trace)


def test_normalize_defaults_hitl_fields():
    trace = normalize_rag_trace({})
    assert trace["retrieval_status"] is None
    assert trace["hitl_prompt"] is None
    assert trace["hitl_options"] == []


def test_normalize_caps_chunks():
    docs = [{"filename": "a.pdf", "text": "x"} for _ in range(20)]
    trace = normalize_rag_trace({"retrieved_chunks": docs})
    assert len(trace["retrieved_chunks"]) == 10


def test_is_hitl_trace_status_and_route():
    assert is_hitl_trace({"retrieval_status": "needs_clarification"})
    assert is_hitl_trace({"route": "scope_select"})
    assert not is_hitl_trace({"route": "rewrite_question"})


def test_route_from_trace():
    assert hitl_route_from_trace({"retrieval_status": "needs_scope_selection"}) == "scope_select"
    assert hitl_route_from_trace({"route": "clarify"}) == "clarify"


def test_compose_question():
    q = compose_question("报销标准是什么", ["技术部的"])
    assert "报销标准是什么" in q and "技术部的" in q


def test_format_hitl_message_with_and_without_options():
    assert format_hitl_message("请补充", None) == "请补充"
    out = format_hitl_message("选一个", ["A", "B"])
    assert "A" in out and "B" in out and "可选方向" in out


def test_detect_scope_select_by_clustering():
    docs = [
        {"filename": "销售.md", "text": "t", "page_number": 1},
        {"filename": "销售.md", "text": "t2", "page_number": 2},
        {"filename": "财务.md", "text": "t3", "page_number": 3},
        {"filename": "财务.md", "text": "t4", "page_number": 4},
    ]
    d = detect_hitl("报销标准", docs, router_model=None)
    assert d.needs_hitl and d.route == "scope_select"
    assert set(d.options) == {"销售", "财务"}


def test_detect_no_hitl_single_cluster():
    docs = [
        {"filename": "销售.md", "text": "t", "page_number": 1},
        {"filename": "销售.md", "text": "t2", "page_number": 2},
    ]
    d = detect_hitl("销售提成", docs, router_model=None)
    assert not d.needs_hitl


class _StubRouter:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return type("R", (), {"content": self.reply})()


def test_detect_clarify_with_router():
    docs = [{"filename": "销售.md", "text": "t", "page_number": 1}]
    router = _StubRouter("请问指的是哪个部门的报销标准？")
    d = detect_hitl("报销标准", docs, router_model=router)
    assert d.needs_hitl and d.route == "clarify"
    assert d.retrieval_status == "needs_clarification"
    assert d.prompt


def test_detect_no_clarify_when_router_says_no():
    docs = [{"filename": "销售.md", "text": "t", "page_number": 1}]
    router = _StubRouter("NO")
    d = detect_hitl("销售提成是多少", docs, router_model=router)
    assert not d.needs_hitl


def test_detect_no_knowledge_when_empty_docs():
    d = detect_hitl("任意问题", [], router_model=None)
    assert d.retrieval_status == "no_knowledge"
    assert not d.needs_hitl
