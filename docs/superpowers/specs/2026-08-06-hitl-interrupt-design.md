# ChatCat：HITL 澄清/范围选择 + 持久化笔记 + 会话标题（LangGraph interrupt 统一图）

- **日期**：2026-08-06
- **状态**：设计已批准，待实现计划
- **范围**：将外部「进化版 agent + storage」片段中的 4 个特性移植进 ChatCat：HITL 澄清/范围选择、持久化笔记、自动会话标题、请求上下文收敛

---

## 1. 背景

当前 ChatCat 的对话流程是两层：

```
create_agent (LangChain 工具循环)
  └─ search_knowledge_base 工具
       └─ run_rag_graph (LangGraph: retrieve→grade→rewrite→expand)
```

问题：

1. **无 HITL**。System Prompt 已预告 `NEEDS_CLARIFICATION` / `NEEDS_SCOPE_SELECTION` / `NO_KNOWLEDGE`（agent.py:43-44），但 RAG 层与 Agent 层从未真正产出这些状态；`search_knowledge_base` 只返回文本。
2. **`interrupt()` 与工具调用不兼容**。LangGraph 的 `interrupt()` 只能在图节点内调用，在工具调用线程里会报错（contextvar 未设置）。因此想用原生 HITL，必须把「知识检索」从 agent 工具提升为「自有图的节点」——这是统一图的前提，不是可选重构。
3. 外部片段一（进化版 agent）用手写的 `pending_hitl` 字典 + `resume_state` 透传 + `answers[]` 累加实现 HITL，约 200 行状态机，且依赖 `metadata_json` 承载运行态；片段二存储层是退化版（全删全插）。

## 2. 关键决策

| # | 决策 | 结论 |
|---|---|---|
| 1 | 移植范围 | 全部：HITL + 持久化笔记 + 会话标题 + 请求上下文收敛 |
| 2 | HITL 触发判定 | 混合：router LLM 判 `clarify`，聚类启发式判 `scope_select`，检索为空→`no_knowledge` |
| 3 | 上下文窗口 | **保留** token 级 `_manage_context_window`（摘要 + `superseded_by`），`persistent_note` 作为独立工作记忆叠加 |
| 4 | HITL 机制 | LangGraph 原生 `interrupt` / `Command(resume=...)`，**淘汰**手写 `pending_hitl` 状态机与 `resume_state` 透传 |
| 5 | Checkpointer | **PostgresSaver**（新增依赖 `langgraph-checkpoint-postgres`），复用现有 PostgreSQL |
| 6 | 存储层 | 保留现有 append-only `ConversationStorage`，仅新增 `load_with_meta` + trace 归一化；**不**采用片段二全删全插 |

已核实的运行环境（WSL venv）：

- `langgraph 1.1.3`、`langchain 1.2.13`、`langgraph-checkpoint 4.0.1`
- `from langgraph.types import interrupt, Command` ✅（1.x 中 `Command` 位于 `langgraph.types`，不在 `langgraph.graph`）
- `InMemorySaver`/`MemorySaver` 已装；`SqliteSaver`/`PostgresSaver` 需额外依赖，选 PostgresSaver

## 3. 统一 StateGraph

```
backend/rag/chat_graph.py

  prepare_context → route
      ├─ weather  → generate → END
      ├─ retrieve → check_hitl → generate → END
      │                 │ 无 HITL
      │                 └─ interrupt() 命中 → resume 时 answers.append(answer)、
      │                     question=compose(...)，边指回 retrieve 重跑（§4）
      └─ generate（闲聊）→ END
```

- **`route`**：规则优先识别天气（关键词）+ router LLM 判定「知识库 / 闲聊」。闲聊与天气不空跑 Milvus。写入 `state["route"]`。
- **`retrieve`**：复用现有 `run_rag_graph`（grade / rewrite / expand / auto-merge 全保留），从「agent 工具」变为「图节点」，返回 `docs` / `context` / `rag_trace`。
- **`check_hitl`**：混合判定 → 命中则 `interrupt(...)`。**resume 后改写 question 并指回 `retrieve` 重跑**（见 §4）。
- **`generate`**：按 `state["route"]` 分支——
  - `route == "retrieve"`：必须基于 docs 回答（System 强制引用 `[1][2]`）；**docs 为空 → 输出「知识库无可靠信息」**（与 §5 `no_knowledge` 一致，不得自由编造）；
  - `route == "weather"`：基于天气工具结果回答；
  - `route == "generate"`：普通闲聊，自由回答（无 docs 是正常的）。
- **`weather`**：直接复用现有 `get_current_weather` 逻辑，不走工具循环。

## 4. HITL 闭环（interrupt 语义，含关键修正）

`interrupt()` 语义：`answer = interrupt(value)` 在首次执行时抛出 `GraphInterrupt` 暂停图；resume 时该调用**返回** `Command(resume=...)` 传入的值，节点在 `interrupt()` 之后继续执行。

**关键修正**：`check_hitl` 若放在 `retrieve` 之后直接 `interrupt()`，resume 会从 `check_hitl` 之后继续 → 直接 `generate`，`retrieve` 不重跑，导致 generate 用的是**歧义问题检索出的旧 docs**。因此：

```python
def check_hitl(state):
    if len(state["answers"]) >= MAX_HITL_ROUNDS:        # 防死循环，默认 3
        return {"next": "generate"}
    decision = detect_hitl(state)                        # 混合判定（§5）
    if not decision.needs_hitl:
        return {"next": "generate"}
    answer = interrupt({                                 # 暂停，向前端发 hitl_request
        "route": decision.route,                          # "clarify" | "scope_select"
        "prompt": decision.prompt,
        "options": decision.options,
    })
    # ── 以下仅在 resume 后执行 ──
    answers = [*state["answers"], answer]
    return {"answers": answers, "question": compose_question(state, answers), "next": "retrieve"}
```

条件边：`next == "retrieve"` → `retrieve`；否则 → `generate`。嵌套反问（补了还不够）由 `check_hitl → retrieve → check_hitl` 循环自然处理，`answers[]` 存图状态、随 checkpoint 跨轮保留，**无需手写 `_build_hitl_resume_query`**。

图状态：

```python
class ChatState(TypedDict):
    user_text: str
    original_question: str
    question: str                  # 原始问题 + answers 合成
    answers: list[str]             # HITL 跨轮累积补充
    persistent_note: str
    history: list                  # 每轮由编排层注入（含 token 裁剪/摘要）
    route: str                     # "retrieve" | "weather" | "generate"
    docs: list
    context: str
    rag_trace: dict
    response: str
```

## 5. HITL 触发判定（混合）

`backend/rag/hitl_detect.py`，对 `retrieve` 结果判定：

| 判定 | 触发条件 | 产出 |
|---|---|---|
| `no_knowledge` | 检索 0 条 | `retrieval_status=no_knowledge`，`generate` 输出「知识库无可靠信息」 |
| `needs_clarification` | **router LLM**（复用 `_get_router_model`，单次调用）判 query 歧义/缺槽位 | `hitl_prompt`（LLM 生成的追问） |
| `needs_scope_selection` | **启发式**：top 结果按 `filename` 聚类，≥2 簇且簇内条数 ≥ 阈值 | `hitl_options`（簇标签方向描述） |

降级策略：router LLM 调用失败 → 仅走启发式；启发式也未触发 → 走原流程，**绝不因 HITL 检测失败阻断对话**。`MAX_HITL_ROUNDS = 3`。

## 6. Checkpointer 与线程策略

- `PostgresSaver`，装配在 `backend/infra/` 新模块，复用 `DATABASE_URL`；应用启动时 `saver.setup()` 建 checkpoint 表。
- **thread_id = `f"{user_id}:{session_id}:{question_id}"`**（每问题一线程）：
  - question_id 在**同一问题的多轮 HITL 内稳定**，新问题新生成；
  - 新问题 = 新线程 = 干净状态，**不依赖 checkpoint 删除 API**（1.x 中该 API 不稳定）；
  - 中断后 `metadata_json["pending_hitl_thread_id"]` 指向当前线程；resume 用同一 thread_id + `Command(resume=user_text)`；
  - 放弃反问 = 清除该 metadata 键（前端可选加「换个话题」按钮触发）。
- 安全：thread_id 含 user_id 命名空间，API 层校验会话归属，防止跨用户 resume。

## 7. 编排层（agent.py 重写为薄编排）

`chat_with_agent` / `chat_with_agent_stream` **签名不变**（api.py 无需改动），内部：

1. 加载 metadata + 历史 → `_manage_context_window` 裁剪；
2. `metadata_json["pending_hitl_thread_id"]` 存在 → `graph.astream(Command(resume=user_text, update={"history":…, "persistent_note":…}), config)`；否则 → `graph.astream(initial_state, config={新 question thread})`；
3. `stream_mode=["messages", "custom"]`：`custom` 事件 → `rag_step` SSE（**图跑在 async 端点，tools.py 跨线程桥接整块删除**；仅 Milvus/embedding 等同步调用用 `run_in_executor` 包裹）；`messages` 事件 → 内容 token；
4. 流结束且 `graph.get_state(config)` 有 pending interrupt → 发 `hitl_request` SSE + `[DONE]`，把追问写为一条 AI ChatMessage（历史可见），存 `pending_hitl_thread_id`；
5. 正常结束 → 更新 `persistent_note`（`fast_model`，executor 中跑）、首条消息生成 `session_title` 并写 `metadata_json["title"]`、写 ChatMessage（human + AI + `rag_trace`）。

`/chat`（非流式）端点保持可用：`asyncio.run(graph.ainvoke(...))` 或 executor 内跑，HITL 时在响应中返回 `hitl_request` 语义字段。

## 8. 存储 / 历史 / 笔记 / 标题

| 数据 | 存放 | 说明 |
|---|---|---|
| 图运行态（question/answers/docs/interrupt） | **checkpoint**（PostgresSaver，按 question 维度） | `Command(resume)` 精确续跑 |
| 会话历史（含摘要） | **ChatMessage + metadata_json** | UI 数据源；每轮由编排层重新注入 `history`，图上不累积历史 |
| `persistent_note` / `title` / `pending_hitl_thread_id` | **metadata_json** | 会话级状态 |
| `fast_model` | env `LLM_FAST_MODEL` | 笔记维护用廉价快模型 |

- `ConversationStorage`：保留 append-only + `superseded_by` + `token_count` + `message_index` + `_CACHE_VERSION`；新增 `load_with_meta(user, session) → (messages, metadata)`；`save` / `get_session_messages` 对 `rag_trace` 走 `normalize_rag_trace`。
- `normalize_rag_trace(raw)`（位于 `backend/rag/hitl_detect.py`，与检测逻辑同模块，供 agent 编排层与存储层共用）：兼容存量旧 trace（旧 `route: generate_answer/rewrite_question` 不得误判为 HITL 路由），保证 `retrieval_status` / `route` / `hitl_prompt` / `hitl_options` 有安全默认值，`retrieved_chunks` 截断到前 N 条防 JSON 膨胀。

## 9. 前端

- `types/chat.ts`：新增 `HitlRequest`（`route` / `prompt` / `options` / `original_question`）。
- `stores/chat.ts`：SSE 新增 `hitl_request`（存到 botMsg、渲染追问 + 选项按钮）与 `session_title`（更新 `sessions` 中对应 title）。
- 新组件 `Chat/HitlMessage.vue`：渲染追问文本 + 选项按钮；点击后照常发消息，**前端无需感知 HITL 状态机**（后端靠 `pending_hitl_thread_id` 自动识别续答）。

## 10. 错误处理与边界

- **中断后新问题**：有 `pending_hitl_thread_id` 时默认按续答处理；「换个话题」按钮清除该键（可选）。
- **嵌套 HITL**：`MAX_HITL_ROUNDS` 上限，超出强制 `generate`。
- **LLM 降级**：router/检测失败不阻断，走启发式或原流程。
- **服务重启**：checkpoint 持久化在 Postgres，未答完的反问跨重启存活。
- **跨用户安全**：thread_id 含 user 命名空间 + API 层归属校验。

## 11. 测试

- `hitl_detect.py`、`chat_graph.py`（interrupt → `Command(resume)` 闭环）可纯单元测试，无需 mock 整只 agent——统一图带来的可测性红利。
- `normalize_rag_trace` 兼容旧 trace 用例、`_is_hitl_trace` 边界用例。
- 前端 `chat.ts` 的 `hitl_request` / `session_title` 分支加 vitest。
- LLM / Milvus 集成部分手动冒烟。

## 12. 文件改动清单

| 文件 | 动作 |
|---|---|
| `pyproject.toml` | + `langgraph-checkpoint-postgres` |
| `backend/rag/chat_graph.py` | **新增**：统一 StateGraph + PostgresSaver 装配 |
| `backend/rag/hitl_detect.py` | **新增**：混合判定（router LLM + 聚类） |
| `backend/infra/checkpointer.py` | **新增**：PostgresSaver 连接装配 + `setup()` |
| `backend/agent/agent.py` | 重写为编排层；保留存储 + token 窗口；删手写 HITL 状态机 + `create_agent` |
| `backend/agent/tools.py` | 删 `search_knowledge_base`；保留天气 |
| `backend/rag/rag_pipeline.py` | 基本不动，被 `retrieve` 节点复用 |
| `backend/api/api.py` | 不变（编排层签名不变） |
| 前端 `chat.ts` / `types/chat.ts` / `Chat/HitlMessage.vue` | 处理 `hitl_request` / `session_title` |
| `.env.example`（若有） | + `LLM_FAST_MODEL` |

## 13. 迁移与回滚

- checkpoint 表由 `saver.setup()` 幂等创建，无需手工迁移。
- 存量会话：无 `pending_hitl_thread_id`，走新问题路径，天然兼容；旧 `rag_trace` 经 `normalize_rag_trace` 归一化。
- 回滚：恢复旧 agent.py / tools.py，checkpoint 表可保留（不阻塞）。
