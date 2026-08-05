# HITL 澄清/范围选择 + 持久化笔记 + 会话标题 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 HITL（澄清/范围选择）、持久化笔记、自动会话标题移植进 ChatCat，用 LangGraph 原生 `interrupt`/`Command(resume)` 取代手写状态机，统一为单一 StateGraph。

**Architecture:** 用 `backend/rag/chat_graph.py` 的单一 StateGraph（prepare_context → route → retrieve/check_hitl/generate）替代现有的 `create_agent` + `search_knowledge_base` 工具两层结构。HITL 用 `interrupt()` 暂停、`Command(resume=...)` 续跑，状态存 PostgresSaver checkpoint（thread_id = `{user_id}:{session_id}`）。编排层（`agent.py`）保留签名不变，驱动图 + 笔记 + 标题 + ChatMessage 持久化。图节点全为同步（generate 用 `model.stream`），流式路径在线程里跑 `graph.stream`，通过 `get_stream_writer()` 的 `custom` 事件推送 RAG 步骤与内容 token。

**Tech Stack:** Python 3.12, langgraph 1.1.3（`langgraph.types.interrupt`/`Command`、`langgraph.config.get_stream_writer`、`langgraph.checkpoint.postgres.PostgresSaver`）, langchain 1.2.13, FastAPI, Vue 3 + Pinia + vitest, pytest。

## Global Constraints

- 后端命令一律在 WSL 内执行（`wsl -d Ubuntu-20.04 bash -ic "cd /home/mspace/code/ChatCat && ..."`）。前端命令同理。
- 关键 import（已实测，勿改动）：`from langgraph.types import interrupt, Command`；`from langgraph.config import get_stream_writer`；`Command` 不在 `langgraph.graph`。
- `interrupt` 通过 `astream`/`stream` 时**流静默结束**（不产生 `__interrupt__` 事件）；中断检测一律用 `graph.get_state(config)`：`snap.tasks[0].interrupts[0].value`。
- **resume 时被中断的节点会从头重跑**：`detect_fn` 每轮 HITL 会多执行一次（重入那次），检测逻辑必须对同输入确定（router temperature=0）才能保证 resume 到达 `interrupt()` 并返回用户补充。
- 图节点全部为**同步**函数（含 `generate` 用 `model.stream`），避免 async 节点在 executor 里的 contextvar 丢失问题。
- 保留现有 append-only `ConversationStorage`、token 级 `_manage_context_window`、`_CACHE_VERSION="v2"`；不采用片段二全删全插。
- 保留 `backend/rag/rag_pipeline.py` 的 `run_rag_graph` 原样，`retrieve` 节点直接调用它。
- 旧 trace 的 `route` 值 `generate_answer`/`rewrite_question` 不得被误判为 HITL 路由。
- ruff/black 已配置（pyproject `[tool.ruff]` line-length=120）；后端改完跑 `uv run ruff check backend && uv run ruff format backend`。

---

### Task 1: 后端依赖 + pytest 测试基建

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: `uv run pytest` 在项目根可跑通（backend 测试）；`langgraph-checkpoint-postgres` + `psycopg[binary]` 已入依赖。

- [ ] **Step 1: 加依赖**

在 `pyproject.toml` 的 `dependencies` 加：
```toml
"langgraph-checkpoint-postgres>=2.0.0",
"psycopg[binary]>=3.1.0",
```
在 `[project.optional-dependencies].dev` 加：
```toml
"pytest>=8.0.0",
"pytest-asyncio>=0.23.0",
```
（`langgraph-checkpoint-postgres` 的 `PostgresSaver` 需要 psycopg v3 连接；现有 `psycopg2-binary` 只用于 SQLAlchemy，保留。）

- [ ] **Step 2: 配置 pytest**

在 `pyproject.toml` 追加：
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: 写冒烟测试**

`tests/test_smoke.py`：
```python
def test_smoke():
    assert 1 + 1 == 2
```

`tests/conftest.py`（保证纯函数测试不触发外部连接；各测试如需真实服务自行设 env 标记）：
```python
import os

os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("MILVUS_HOST", "localhost")
```

- [ ] **Step 4: 安装并跑通**

Run（WSL）: `cd /home/mspace/code/ChatCat && uv sync && uv run pytest -q`
Expected: `1 passed`。

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "chore: add pytest harness and langgraph-checkpoint-postgres deps"
```

---

### Task 2: PostgresSaver checkpointer 装配

**Files:**
- Create: `backend/infra/checkpointer.py`
- Modify: `backend/core/app.py`（startup 调 `init_checkpointer()`）

**Interfaces:**
- Produces: `get_checkpointer() -> PostgresSaver`（进程级单例）、`init_checkpointer() -> None`（建表）。

- [ ] **Step 1: 写工厂**

`backend/infra/checkpointer.py`：
```python
"""PostgresSaver checkpointer 装配（HITL 状态持久化，复用现有 PostgreSQL）。"""
import os
import threading

from langgraph.checkpoint.postgres import PostgresSaver


def _postgres_dsn() -> str:
    dsn = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/chatcat_app",
    )
    # langgraph-checkpoint-postgres 需要 psycopg v3 的 DSN（无 SQLAlchemy 方言前缀）
    return dsn.replace("+psycopg2", "")


_saver: PostgresSaver | None = None
_saver_lock = threading.Lock()


def get_checkpointer() -> PostgresSaver:
    """进程级单例。连接按需建立；连接断开后由 langgraph 内部自动重连。"""
    global _saver
    if _saver is None:
        with _saver_lock:
            if _saver is None:
                import psycopg

                conn = psycopg.connect(_postgres_dsn(), autocommit=True)
                _saver = PostgresSaver(conn)
    return _saver


def init_checkpointer() -> None:
    """幂等创建 checkpoint 表（应用启动时调用）。"""
    get_checkpointer().setup()
```

- [ ] **Step 2: 挂到启动钩子**

`backend/core/app.py` 的 startup 里追加：
```python
from backend.infra.checkpointer import init_checkpointer

@app.on_event("startup")
async def _startup_init_db():
    init_db()
    init_checkpointer()
```

- [ ] **Step 3: 验证**

Run（WSL）: `cd /home/mspace/code/ChatCat && docker compose up -d && uv run python -c "from backend.infra.checkpointer import get_checkpointer, init_checkpointer; init_checkpointer(); s = get_checkpointer(); print('ok', type(s).__name__); print('has delete_thread:', hasattr(s, 'delete_thread'))"`
Expected: `ok PostgresSaver` 且 `has delete_thread: True`（若 Postgres 未启动会报连接错误，先 `docker compose up -d` 等健康）。

- [ ] **Step 4: 提交**

```bash
git add backend/infra/checkpointer.py backend/core/app.py
git commit -m "feat: PostgresSaver checkpointer wiring for HITL state"
```

---

### Task 3: HITL 检测模块（纯函数，TDD）

**Files:**
- Create: `backend/rag/hitl_detect.py`
- Test: `tests/test_hitl_detect.py`

**Interfaces:**
- Produces:
  - `MAX_HITL_ROUNDS = 3`
  - `HITL_STATUSES = {"needs_clarification", "needs_scope_selection", "no_knowledge"}`
  - `HITL_ROUTES = {"clarify", "scope_select"}`
  - `normalize_rag_trace(raw: dict | None) -> dict`（旧 trace 兼容 + 安全默认值 + chunks 截断）
  - `is_hitl_trace(trace: dict | None) -> bool`
  - `hitl_route_from_trace(trace: dict) -> str`（`"clarify"` | `"scope_select"`）
  - `compose_question(original: str, answers: list[str]) -> str`
  - `format_hitl_message(prompt: str, options: list[str] | None = None) -> str`
  - `class HitlDecision`（dataclass：`needs_hitl: bool, route: str, prompt: str, options: list[str], retrieval_status: str`）
  - `detect_hitl(question: str, docs: list[dict], router_model=None) -> HitlDecision`（混合判定）

- [ ] **Step 1: 写失败测试**

`tests/test_hitl_detect.py`：
```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/mspace/code/ChatCat && uv run pytest tests/test_hitl_detect.py -q`
Expected: FAIL（`ModuleNotFoundError: backend.rag.hitl_detect`）。

- [ ] **Step 3: 实现**

`backend/rag/hitl_detect.py`：
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/mspace/code/ChatCat && uv run pytest tests/test_hitl_detect.py -q`
Expected: PASS（全部）。

- [ ] **Step 5: 提交**

```bash
git add backend/rag/hitl_detect.py tests/test_hitl_detect.py
git commit -m "feat: HITL detection helpers + rag_trace normalization"
```

---

### Task 4: 统一 StateGraph（TDD，DI 注入替身）

**Files:**
- Create: `backend/rag/chat_graph.py`
- Test: `tests/test_chat_graph.py`

**Interfaces:**
- Produces:
  - `class ChatState(TypedDict)`（字段见下）
  - `build_chat_graph(*, retrieve_fn=None, detect_fn=None, generate_fn=None, weather_fn=None, checkpointer=None)`：全部注入点默认为真实实现；`checkpointer` 默认 `InMemorySaver()`（生产由编排层传 `get_checkpointer()`）。
  - 节点全部**同步**；`generate` 用 `model.stream` + `get_stream_writer()` 发 `{"type":"content","content":str}`；`retrieve` 发 `{"type":"rag_step","step":{...}}`。
  - `run_rag_graph` 由默认 `retrieve_fn` 懒加载调用（保持 `rag_pipeline.py` 不动）。

**ChatState 字段：**
```python
class ChatState(TypedDict):
    user_text: str
    original_question: str
    question: str
    answers: list[str]
    persistent_note: str
    history: list[dict]      # [{"role": "user"|"assistant"|"system", "content": str}]
    mode: str                # "" | "retrieve" | "weather" | "generate"
    phase: str               # 条件边信号："" | "retrieve" | "weather" | "generate"
    docs: list[dict]
    context: str
    rag_trace: dict
    weather_result: str
    response: str
```

**图拓扑与条件边：**
```
START → prepare_context → route ──(phase)──► weather → generate → END
                                    └─(phase)──► retrieve → check_hitl ──(phase)──► generate → END
                                                 (check_hitl 命中时 interrupt()；resume 后 phase="retrieve" 回环)
```

- [ ] **Step 1: 写失败测试**

`tests/test_chat_graph.py`（走**真实 `check_hitl` 节点**，注入 `detect_fn` 触发 interrupt）：
```python
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


def _fake_generate(state):
    return {"response": f"回答:{state['question']}"}


def _always_no_hitl(question, docs, router_model=None):
    return HitlDecision(needs_hitl=False)


def _hitl_once(decision):
    # LangGraph resume 时被中断节点从头重跑，detect 每轮 HITL 多执行一次（第 2 次为重入）
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
        # 两轮 HITL × 每次重入，共 4 次命中
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/mspace/code/ChatCat && uv run pytest tests/test_chat_graph.py -q`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现图**

`backend/rag/chat_graph.py`：
```python
"""统一对话 StateGraph：prepare_context → route → {weather,retrieve→check_hitl} → generate。

HITL 用 LangGraph 原生 interrupt/Command(resume)；节点全同步，RAG 步骤与内容
通过 get_stream_writer() 发 custom 事件，供编排层转 SSE。
"""
from typing import Callable, Optional, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from backend.rag.hitl_detect import (
    HitlDecision,
    compose_question,
    format_hitl_message,
    hitl_route_from_trace,
    is_hitl_trace,
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
    writer({"type": "rag_step", "step": {"icon": "🔍", "label": "正在检索知识库...", "detail": f"查询: {state['question'][:50]}"}})
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


def weather_node(state: ChatState) -> dict:
    from backend.agent.tools import get_current_weather

    result = get_current_weather(state["question"])
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
        messages = [HumanMessage(content="你是天气助手，基于天气数据简洁回答。"), *_history_to_messages(state["history"]), human]
    else:
        messages = []
        if state["persistent_note"]:
            messages.append(SystemMessage(content=f"【对话持久化笔记（你的工作记忆）】\n{state['persistent_note']}\n请参考以上笔记保持对话连贯性。"))
        messages.extend(_history_to_messages(state["history"]))
        messages.append(HumanMessage(content=state["question"]))

    model = _get_chat_model()
    full = ""
    try:
        for chunk in model.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if isinstance(content, list):
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else (b if isinstance(b, str) else "")
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/mspace/code/ChatCat && uv run pytest tests/test_chat_graph.py -q`
Expected: PASS。重点断言：
- `test_hitl_interrupt_blocks_and_resumes`：第一次 `invoke` 后 `get_state().tasks[0].interrupts[0].value["route"] == "clarify"`；`Command(resume="技术部的")` 后 response 含补充内容。
- `test_no_knowledge`：docs 空 → response == NO_KNOWLEDGE。

- [ ] **Step 5: 验证 custom 事件在 sync stream 下发**

追加一个临时断言测试（保留为正式用例）：
```python
def _fake_retrieve_with_step(state):
    from langgraph.config import get_stream_writer
    get_stream_writer()({"type": "rag_step", "step": {"icon": "🔍", "label": "检索"}})
    return {"docs": [{"filename": "销售.md", "text": "t", "page_number": 1}], "context": "t", "rag_trace": {"retrieval_status": "ok"}}


def test_custom_events_from_sync_stream():
    graph = build_chat_graph(retrieve_fn=_fake_retrieve_with_step, generate_fn=lambda s: s, checkpointer=InMemorySaver())
    events = [d for _k, d in graph.stream(
        {"user_text": "hi", "original_question": "", "question": "", "answers": [],
         "persistent_note": "", "history": [], "mode": "", "phase": "", "docs": [], "context": "",
         "rag_trace": {}, "weather_result": "", "response": ""},
        config={"configurable": {"thread_id": "t3"}}, stream_mode=["custom"])]
    assert any(ev.get("type") == "rag_step" for ev in events)
```
Run: `uv run pytest tests/test_chat_graph.py::test_custom_events_from_sync_stream -q`
Expected: PASS（确认 `get_stream_writer` 在 sync `graph.stream` 的节点内可用）。

- [ ] **Step 6: 提交**

```bash
git add backend/rag/chat_graph.py tests/test_chat_graph.py
git commit -m "feat: unified interruptible chat StateGraph with DI"
```

---

### Task 5: 存储层加 `load_with_meta` + trace 归一化

**Files:**
- Modify: `backend/agent/agent.py`（`ConversationStorage`）
- Test: `tests/test_storage_meta.py`（集成，无 DB 则 skip）

**Interfaces:**
- Produces: `ConversationStorage.load_with_meta(user_id, session_id) -> tuple[list, dict]`；`save`/`get_session_messages` 对 `rag_trace` 走 `normalize_rag_trace`。

- [ ] **Step 1: 写失败测试**

`tests/test_storage_meta.py`：
```python
import os
import pytest

from backend.agent.agent import storage

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.path.exists("/tmp/chatcat_test_pg"),
    reason="需要本地 PostgreSQL + Redis",
)


def test_load_with_meta_roundtrip():
    uid, sid = "test_user", "test_session"
    # 无账号会被 storage 忽略，这里用不存在的账号验证返回形状
    messages, meta = storage.load_with_meta(uid, sid)
    assert isinstance(messages, list)
    assert isinstance(meta, dict)
```
（该用例主要验证签名与返回形状；真实持久化行为由后续手动冒烟覆盖。若本地无 Postgres，`pytest -k storage` 自动 skip。）

- [ ] **Step 2: 实现**

在 `backend/agent/agent.py` 的 `ConversationStorage` 内：

`load_with_meta`（放在 `load` 之后）：
```python
def load_with_meta(self, user_id: str, session_id: str) -> tuple[list, dict]:
    """加载对话消息及会话元数据（标题、持久化笔记、HITL 状态等）。"""
    messages = self.load(user_id, session_id)
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == user_id).first()
        if not user:
            return messages, {}
        session = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
            .first()
        )
        if not session:
            return messages, {}
        return messages, dict(session.metadata_json or {})
    finally:
        db.close()
```

在 `save` 里，序列化 `rag_trace` 前归一化（替换两处 `rag_trace = ...` 赋值）：
```python
from backend.rag.hitl_detect import normalize_rag_trace  # 文件顶部导入
...
rag_trace = normalize_rag_trace(extra.get("rag_trace"))
```

在 `get_session_messages` 里：
```python
"rag_trace": normalize_rag_trace(row.rag_trace),
```

- [ ] **Step 3: 跑测试 + lint**

Run: `cd /home/mspace/code/ChatCat && uv run pytest -q && uv run ruff check backend/agent/agent.py`
Expected: 现有测试 + 新增用例通过；ruff 无错误。

- [ ] **Step 4: 提交**

```bash
git add backend/agent/agent.py tests/test_storage_meta.py
git commit -m "feat: storage load_with_meta + rag_trace normalization"
```

---

### Task 6: 编排层重写（agent.py 驱动图 + 笔记 + 标题 + 持久化）

**Files:**
- Modify: `backend/agent/agent.py`（保留 `ConversationStorage`、`count_tokens`、`_manage_context_window`、`summarize_old_messages`；替换 `chat_with_agent`/`chat_with_agent_stream` 内部实现；删 `create_agent_instance`/`agent`/`get_system_prompt`）
- Modify: `backend/agent/tools.py`（`emit_rag_step` 改用 `get_stream_writer`；删 `search_knowledge_base`、跨线程全局）
- Modify: `backend/api/api.py`（若 `chat_with_agent` 保持同步则不改；见下）
- Modify: `backend/core/app.py`（如需挂 `init_checkpointer`，已在 Task 2 完成）

**Interfaces:**
- Consumes: `build_chat_graph()`、`ChatState`、`get_checkpointer()`、`normalize_rag_trace`、`is_hitl_trace`、`format_hitl_message`。
- Produces: `chat_with_agent(user_text, user_id, session_id) -> dict`（含 `response`、`rag_trace`、可选 `hitl`）、`chat_with_agent_stream(...)` async generator（SSE 事件：`rag_step`/`content`/`trace`/`hitl_request`/`session_title`）。

**关键设计（实现者必读）：**
- `chat_with_agent`（同步）与 `chat_with_agent_stream`（async）共享同一套「跑图 + 收事件 + 收尾持久化」逻辑。同步版直接 `graph.stream(..., stream_mode=["custom"])`；流式版用 `asyncio.to_thread(lambda: list(graph.stream(...)))` 把整轮跑进线程，把 custom 事件经 `asyncio.Queue` 转发给 async generator（与现有 `_agent_worker` + `output_queue` 架构一致）。
- **中断检测**：一轮流结束后 `graph.get_state(cfg)`，若 `tasks[0].interrupts` 存在 → 这是 HITL 暂停：发 `hitl_request` SSE、把追问写为一条 AI 消息持久化、结束本轮（**不**更新笔记）。否则正常完成 → 更新笔记 + 持久化 human/AI。
- **线程/thread_id**：`cfg = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}`（每会话一线程）。新消息时先 `get_state`：有 pending interrupt → `Command(resume=user_text)`；否则 → 注入 `{user_text, persistent_note, history}` 走新轮（`prepare_context` 会重置轮内字段）。
- **history 注入**：`history = [{"role": ..., "content": str(m.content)} for m in trimmed_messages[:-1]]`（含摘要 role=system）。
- **`fast_model`**：新 env `LLM_FAST_MODEL`（缺省 `LLM_MODEL`），供笔记维护。

- [ ] **Step 1: `tools.py` 重写 `emit_rag_step` + 删工具**

`backend/agent/tools.py`：
- 删除模块级 `_last_rag_context`/`_global_rag_context`/`_rag_step_queue`/`_rag_step_loop` 及相关函数（`_set_last_rag_context`、`get_last_rag_context`、`reset_tool_call_guards`、`set_rag_step_queue`）。
- `emit_rag_step` 改为：
```python
def emit_rag_step(icon: str, label: str, detail: str = "") -> None:
    """向当前图的 custom 流发送一个 RAG 步骤（图节点内调用有效，否则静默跳过）。"""
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()({"type": "rag_step", "step": {"icon": icon, "label": label, "detail": detail}})
    except Exception:
        pass
```
- 删除 `search_knowledge_base` 工具及 `contextvars`/`requests` 中仅被其使用的部分；保留 `get_current_weather`。
- 保留 `import contextvars` 若 `get_current_weather` 不需要则一并删除（`get_current_weather` 只用 requests）。

- [ ] **Step 2: `agent.py` 重写编排**

在 `backend/agent/agent.py`：
- 删除：`create_agent_instance`、模块级 `agent`/`model`、`get_system_prompt`、`create_agent` 相关 import。新增 `_get_fast_model()`（懒加载 `LLM_FAST_MODEL`）。
- 保留：`count_tokens`、`_manage_context_window`、`summarize_old_messages`、`ConversationStorage`。
- 新增核心流程函数（供 sync/stream 复用）：

```python
def _session_thread_config(user_id: str, session_id: str) -> dict:
    return {"configurable": {"thread_id": f"{user_id}:{session_id}"}}


def _history_dicts(messages: list) -> list[dict]:
    out = []
    for m in messages:
        role = "system" if m.type in ("system", "summary") else ("user" if m.type == "human" else "assistant")
        out.append({"role": role, "content": str(m.content)})
    return out


def _run_graph_turn(input_data: dict, cfg: dict) -> tuple[list[dict], dict, dict]:
    """跑一轮图，返回 (custom_events, final_state, pending_hitl_value)。

    final_state 为 graph.get_state(cfg) 快照；pending_hitl_value 为中断值或 None。
    """
    from backend.rag.chat_graph import build_chat_graph
    from backend.infra.checkpointer import get_checkpointer

    graph = build_chat_graph(checkpointer=get_checkpointer())
    events = list(graph.stream(input_data, config=cfg, stream_mode=["custom"]))
    snap = graph.get_state(cfg)
    pending = None
    if snap.tasks and getattr(snap.tasks[0], "interrupts", None):
        pending = snap.tasks[0].interrupts[0].value
    return [d for _kind, d in events], snap, pending
```

`chat_with_agent`（同步，签名不变）：
```python
def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0

    user_msg = HumanMessage(content=user_text)
    user_msg.additional_kwargs["_token_count"] = count_tokens(user_text)
    messages.append(user_msg)
    messages = _manage_context_window(messages, _get_context_model())

    from backend.infra.checkpointer import get_checkpointer
    from backend.rag.chat_graph import build_chat_graph

    cfg = _session_thread_config(user_id, session_id)
    graph = build_chat_graph(checkpointer=get_checkpointer())
    snap = graph.get_state(cfg)
    pending = bool(snap.tasks and getattr(snap.tasks[0], "interrupts", None))

    if pending:
        input_data = Command(resume=user_text)
    else:
        input_data = {
            "user_text": user_text,
            "persistent_note": persistent_note,
            "history": _history_dicts(messages[:-1]),
        }

    events, final_snap, hitl_value = _run_graph_turn(input_data, cfg)
    full_response = ""
    for ev in events:
        if ev.get("type") == "content":
            full_response += ev.get("content", "")

    if hitl_value is not None:
        prompt = hitl_value.get("prompt", "")
        options = hitl_value.get("options") or []
        hitl_text = format_hitl_message(prompt, options)
        rag_trace = normalize_rag_trace(final_snap.values.get("rag_trace") if hasattr(final_snap, "values") else None)
        messages.append(AIMessage(content=hitl_text))
        extra = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
        # 合并进现有 metadata，避免整体覆盖清掉 persistent_note
        save_meta = dict(metadata)
        title = _title_for(metadata, is_first_message, user_text)
        if title:
            save_meta["title"] = title
        storage.save(user_id, session_id, messages, metadata=save_meta, extra_message_data=extra)
        return {"response": hitl_text, "rag_trace": rag_trace, "hitl": hitl_value}

    rag_trace = normalize_rag_trace(final_snap.values.get("rag_trace") if hasattr(final_snap, "values") else None)
    full_response = full_response or (final_snap.values.get("response") if hasattr(final_snap, "values") else "")
    save_meta = dict(metadata)
    if is_first_message:
        save_meta["title"] = generate_session_title(user_text)
    if _should_update_persistent_note(messages, persistent_note):
        save_meta["persistent_note"] = _update_persistent_note_sync(persistent_note, user_text, full_response, history_messages=messages[:-1] if not persistent_note else None)
    messages.append(AIMessage(content=full_response))
    extra = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, metadata=save_meta, extra_message_data=extra)
    return {"response": full_response, "rag_trace": rag_trace}
```

`chat_with_agent_stream`（async，签名不变）核心：
```python
async def chat_with_agent_stream(user_text, user_id="default_user", session_id="default_session"):
    yield f"data: {json.dumps({'type': 'rag_step', 'step': {'icon': '📨', 'label': '请求已接收，正在准备回答'}})}\n\n"

    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0

    user_msg = HumanMessage(content=user_text)
    user_msg.additional_kwargs["_token_count"] = count_tokens(user_text)
    messages.append(user_msg)
    messages = _manage_context_window(messages, _get_context_model())

    from backend.infra.checkpointer import get_checkpointer
    from backend.rag.chat_graph import build_chat_graph

    cfg = _session_thread_config(user_id, session_id)
    graph = build_chat_graph(checkpointer=get_checkpointer())
    snap = await graph.aget_state(cfg)
    pending = bool(snap.tasks and getattr(snap.tasks[0], "interrupts", None))

    input_data = (
        Command(resume=user_text)
        if pending
        else {
            "user_text": user_text,
            "persistent_note": persistent_note,
            "history": _history_dicts(messages[:-1]),
        }
    )

    output_queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            for kind, data in graph.stream(input_data, config=cfg, stream_mode=["custom"]):
                if kind != "custom":
                    continue
                output_queue.put_nowait(data)
            output_queue.put_nowait(None)
        except Exception as e:
            output_queue.put_nowait({"type": "error", "content": str(e)})
            output_queue.put_nowait(None)

    task = asyncio.create_task(asyncio.to_thread(_worker))

    full_response = ""
    session_title = None
    if is_first_message:
        session_title = generate_session_title(user_text)
        yield f"data: {json.dumps({'type': 'session_title', 'title': session_title, 'session_id': session_id})}\n\n"

    try:
        while True:
            ev = await output_queue.get()
            if ev is None:
                break
            if ev.get("type") == "content":
                full_response += ev.get("content", "")
                yield f"data: {json.dumps(ev)}\n\n"
            elif ev.get("type") == "rag_step":
                yield f"data: {json.dumps(ev)}\n\n"
            elif ev.get("type") == "error":
                yield f"data: {json.dumps(ev)}\n\n"
    except GeneratorExit:
        task.cancel()
        raise
    finally:
        if not task.done():
            task.cancel()

    snap = await graph.aget_state(cfg)
    hitl_value = None
    if snap.tasks and getattr(snap.tasks[0], "interrupts", None):
        hitl_value = snap.tasks[0].interrupts[0].value

    rag_trace = normalize_rag_trace(snap.values.get("rag_trace") if hasattr(snap, "values") else None)
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    if hitl_value is not None:
        hitl_event = {
            "route": hitl_value.get("route"),
            "prompt": hitl_value.get("prompt"),
            "options": hitl_value.get("options") or [],
        }
        yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': hitl_event})}\n\n"
        yield "data: [DONE]\n\n"
        hitl_text = format_hitl_message(hitl_event["prompt"], hitl_event["options"])
        messages.append(AIMessage(content=hitl_text))
        extra = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
        save_meta = dict(metadata)
        if session_title:
            save_meta["title"] = session_title
        storage.save(user_id, session_id, messages, metadata=save_meta, extra_message_data=extra)
        return

    full_response = full_response or snap.values.get("response", "")
    yield "data: [DONE]\n\n"

    save_meta = dict(metadata)
    if session_title:
        save_meta["title"] = session_title
    if _should_update_persistent_note(messages, persistent_note):
        try:
            save_meta["persistent_note"] = update_persistent_note(persistent_note, user_text, full_response, history_messages=messages[:-1] if not persistent_note else None)
        except Exception as e:
            print(f"Update persistent note error: {e}")
    messages.append(AIMessage(content=full_response))
    extra = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, metadata=save_meta, extra_message_data=extra)
```

- [ ] **Step 1.5: 新增辅助函数（fast_model / 笔记 / 标题）**

在 `backend/agent/agent.py` 模块级新增（顶部 `import asyncio` 已有；新增 `from langgraph.types import Command`）：

```python
_fast_model = None


def _get_fast_model():
    """笔记/标题等轻任务用的廉价快模型（LLM_FAST_MODEL，缺省 LLM_MODEL）。"""
    global _fast_model
    if _fast_model is None:
        from langchain.chat_models import init_chat_model

        _fast_model = init_chat_model(
            model=os.getenv("LLM_FAST_MODEL") or os.getenv("LLM_MODEL"),
            model_provider="openai",
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0.0,
        )
    return _fast_model


def _get_context_model():
    """_manage_context_window 摘要用的模型。"""
    return _get_fast_model()


CONTEXT_WINDOW_MESSAGES = 6  # 笔记维护触发阈值（消息条数）


def _should_update_persistent_note(messages: list, current_note: str) -> bool:
    """只在短期上下文真正开始裁剪时才花钱维护笔记。"""
    return bool(current_note) or len(messages) > CONTEXT_WINDOW_MESSAGES


def generate_session_title(user_text: str) -> str:
    compact_title = " ".join(user_text.split()).strip(" \t\r\n。！？!?，,；;：:")
    return compact_title[:16] or "新会话"


def _title_for(metadata: dict, is_first_message: bool, user_text: str) -> str | None:
    """首条消息生成标题；否则沿用已有标题（避免整体覆盖 metadata_json）。"""
    if is_first_message:
        return generate_session_title(user_text)
    return metadata.get("title")


async def update_persistent_note(current_note, user_text, ai_response, history_messages=None):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _update_persistent_note_sync(
            current_note, user_text, ai_response, history_messages=history_messages
        ),
    )


def _update_persistent_note_sync(current_note, user_text, ai_response, *, history_messages=None):
    try:
        history_text = ""
        if history_messages:
            lines = [
                f"{'用户' if isinstance(m, HumanMessage) else 'AI'}：{str(m.content)}"
                for m in history_messages
            ]
            history_text = (
                "\n\n▼ 首次建立笔记时需要一并概括的此前对话：\n" + "\n".join(lines) + "\n\n"
            )
        prompt = (
            "你是一个【Context Manager Agent】，负责维护多轮对话中的「持久化笔记」。\n"
            "笔记是模型在有限上下文窗口下的长效工作记忆，记录已解决的问题与关键事实。\n\n"
            "更新规则：\n"
            "1. 将新信息与现有笔记智能合并，不要简单拼接。\n"
            "2. 过滤噪音，控制在 500 字以内，用简明条目输出。\n"
            "3. 若信息冲突，保留最可靠或最新版本。\n\n"
            f"▼ 现有笔记：\n{current_note if current_note else '无'}\n\n"
            f"{history_text}"
            f"▼ 最新一轮对话：\n用户：{user_text}\nAI：{ai_response}\n\n"
            "请直接输出更新后的笔记（纯文本，不要解释或 Markdown 代码块）："
        )
        res = _get_fast_model().invoke([HumanMessage(content=prompt)])
        return (res.content or "").strip()
    except Exception as e:
        print(f"Context Manager Error: {e}")
        return current_note
```

> `generate_session_title`、`_should_update_persistent_note`、`update_persistent_note`、`_update_persistent_note_sync` 均来自片段一，仅将 `fast_model` 换成 `_get_fast_model()`。`_get_context_model()` 供 `_manage_context_window` 用（摘要走快模型，省钱）。

- [ ] **Step 3: 编译与静态检查**

Run: `cd /home/mspace/code/ChatCat && uv run ruff check backend/ && uv run python -c "import backend.agent.agent, backend.rag.chat_graph, backend.agent.tools"`（WSL）
Expected: 无 import 错误；ruff 通过。

- [ ] **Step 4: 提交**

```bash
git add backend/agent/agent.py backend/agent/tools.py backend/api/api.py
git commit -m "feat: rewrite chat orchestration around unified graph (HITL+note+title)"
```

---

### Task 7: 端到端手动冒烟 + 补 `POST /chat/hitl/cancel`

**Files:**
- Modify: `backend/api/api.py`（新增 cancel 端点）

- [ ] **Step 1: 起服务**

Run（WSL）: `cd /home/mspace/code/ChatCat && docker compose up -d && cd frontend && npm run build && cd .. && uv run python -m backend.core.app`（后台）
Expected: FastAPI 起在 :8000，startup 打印 checkpoint 表就绪。

- [ ] **Step 2: 冒烟 1 —— 知识问答**

Run: `curl -s -X POST localhost:8000/chat/stream -H 'Content-Type: application/json' -H "Authorization: Bearer $(curl -s -X POST localhost:8000/auth/login -H 'Content-Type: application/json' -d '{"username":"admin","password":"..."}' | jq -r .access_token)" -d '{"message":"报销标准是什么","session_id":"s1"}' | head -40`
Expected: 看到 `rag_step` 事件 + 正常回答；若检索命中歧义/多簇则看到 `hitl_request`。

- [ ] **Step 3: 冒烟 2 —— HITL 续答**

接上一步：若收到 `hitl_request`，再发 `{"message":"技术部的"}` 同一 `session_id`。
Expected: 返回基于补充的引用式回答（含 `[1]` 引用），流式 `content` 事件；结束后再发新问题不串台。

- [ ] **Step 4: 冒烟 3 —— 天气与闲聊**

Run: `{"message":"武汉今天天气怎么样"}`、`{"message":"你好"}` 分别发。
Expected: 天气走 weather 节点（不走 Milvus 检索）；闲聊走 generate。

- [ ] **Step 5: cancel 端点**

`backend/api/api.py` 追加：
```python
@router.post("/chat/hitl/cancel")
async def cancel_hitl(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """放弃当前 HITL 追问：删除该会话的 checkpoint 线程，下一条消息走新问题。"""
    from backend.infra.checkpointer import get_checkpointer
    from backend.agent.agent import _session_thread_config
    try:
        cfg = _session_thread_config(current_user.username, request.session_id or "default_session")
        get_checkpointer().delete_thread(cfg)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
Run: `curl -s -X POST localhost:8000/chat/hitl/cancel -H 'Content-Type: application/json' -H "Authorization: Bearer $TOKEN" -d '{"session_id":"s1"}'`
Expected: `{"ok":true}`；之后 `{"message":"换个话题：今天天气"}` 走新问题路径。

- [ ] **Step 6: 提交**

```bash
git add backend/api/api.py
git commit -m "feat: cancel pending HITL endpoint"
```

---

### Task 8: 前端 —— SSE 事件处理 + HitlMessage 组件（TDD）

**Files:**
- Create: `frontend/src/utils/sse.ts`, `frontend/src/components/Chat/HitlMessage.vue`
- Modify: `frontend/src/types/chat.ts`, `frontend/src/stores/chat.ts`, `frontend/src/components/Chat/MessageItem.vue`
- Test: `frontend/src/utils/sse.spec.ts`

**Interfaces:**
- Produces: `applySseEvent(msg: Message, data: any) => Message`（纯函数，把 `content`/`rag_step`/`trace`/`error`/`hitl_request`/`session_title` 应用到消息/会话）；`handleHitlReply(text, options)`（点击选项直接发送）。

- [ ] **Step 1: 写失败测试**

`frontend/src/utils/sse.spec.ts`：
```typescript
import { describe, expect, it } from 'vitest';
import { applySseEvent } from './sse';
import type { Message } from '@/types/chat';

describe('applySseEvent', () => {
  it('appends content and clears thinking', () => {
    const msg = { text: '', isUser: false, isThinking: true } as Message;
    const out = applySseEvent(msg, { type: 'content', content: '你好' });
    expect(out.text).toBe('你好');
    expect(out.isThinking).toBe(false);
  });

  it('stores hitl_request', () => {
    const msg = { text: '', isUser: false, isThinking: true } as Message;
    const out = applySseEvent(msg, { type: 'hitl_request', hitl: { route: 'scope_select', prompt: '选一个', options: ['A', 'B'] } });
    expect(out.isThinking).toBe(false);
    expect(out.hitl?.prompt).toBe('选一个');
    expect(out.hitl?.options).toEqual(['A', 'B']);
  });

  it('stores trace', () => {
    const msg = { text: '', isUser: false } as Message;
    const out = applySseEvent(msg, { type: 'trace', rag_trace: { route: 'scope_select' } });
    expect(out.ragTrace?.route).toBe('scope_select');
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/mspace/code/ChatCat/frontend && npx vitest run src/utils/sse.spec.ts`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现纯函数**

`frontend/src/utils/sse.ts`：
```typescript
import type { Message, RagTrace } from '@/types/chat';

export function applySseEvent(msg: Message, data: any): Message {
  switch (data?.type) {
    case 'content':
      return { ...msg, isThinking: false, text: (msg.text || '') + (data.content || '') };
    case 'rag_step':
      return { ...msg, ragSteps: [...(msg.ragSteps || []), data.step], _groupedSteps: (msg as any)._groupedSteps || [] };
    case 'trace':
      return { ...msg, ragTrace: (data.rag_trace as RagTrace) || null };
    case 'error':
      return { ...msg, isThinking: false, text: (msg.text || '') + `\n[Error: ${data.content}]` };
    case 'hitl_request':
      return {
        ...msg,
        isThinking: false,
        hitl: data.hitl,
        text: msg.text || '',
      };
    default:
      return msg;
  }
}
```

`frontend/src/types/chat.ts` 追加：
```typescript
export interface HitlRequest {
  route?: 'clarify' | 'scope_select';
  prompt: string;
  options?: string[];
}
export interface Message { /* 原有 */ hitl?: HitlRequest | null; }
```

- [ ] **Step 4: 接入 store + 组件**

`stores/chat.ts`：
- `handleSend` 中把 SSE 分支替换为 `requestMessages[botMsgIdx] = applySseEvent(requestMessages[botMsgIdx], data);`（覆盖 `content`/`trace`/`rag_step`/`error`/`hitl_request`）。
- 新增 `session_title` 分支：更新 `sessionStore.sessions` 对应 title。

`HitlMessage.vue`：渲染 `msg.hitl.prompt` + 选项按钮；点击选项或文本框发送时照常 `handleSend()`（后端靠 checkpoint 自动识别续答）。在 `MessageItem.vue` 中 `v-if="msg.hitl"` 渲染。

- [ ] **Step 5: 跑前端测试 + 构建**

Run: `cd /home/mspace/code/ChatCat/frontend && npm test && npm run build`
Expected: vitest 全过；`vue-tsc + vite build` 无类型错误。

- [ ] **Step 6: 提交**

```bash
git add frontend/src
git commit -m "feat(frontend): handle hitl_request/session_title SSE + HitlMessage component"
```

---

### Task 9: 收尾 —— 删除死代码与回归

**Files:**
- Modify: `backend/agent/agent.py`（确认无残留 `create_agent`/`search_knowledge_base` 引用）、`backend/agent/tools.py`
- Test: 全量回归

- [ ] **Step 1: 全量 lint + 测试**

Run（WSL）: `cd /home/mspace/code/ChatCat && uv run ruff check backend && uv run ruff format backend --check && uv run pytest -q && cd frontend && npm test && npm run build`
Expected: 全绿。

- [ ] **Step 2: grep 确认无残留**

Run: `cd /home/mspace/code/ChatCat && grep -rn "create_agent\|search_knowledge_base\|get_last_rag_context\|set_rag_step_queue" backend/`
Expected: 无输出（或仅文档注释）。

- [ ] **Step 3: 更新 CLAUDE.md**

在 `Architecture (current)` 与 `Critical gotchas` 中补记：对话流程为统一 `chat_graph.py`、HITL 用 LangGraph interrupt + PostgresSaver checkpoint、`LLM_FAST_MODEL` env、`POST /chat/hitl/cancel`。

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for unified chat graph + HITL"
```
