# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Backend
uv sync
uv run python -m backend.core.app
# or
uv run uvicorn backend.core.app:app --host 0.0.0.0 --port 8000 --reload

# Backend checks
uv run ruff check backend
uv run ruff format backend --check
uv run pytest -q   # 21 tests in tests/

# Frontend dev (proxies to backend:8000)
cd frontend && npm install && npm run build
cd frontend && npm run dev  # Vite on :3000
cd frontend && npm run build

# Infrastructure
docker compose up -d  # PostgreSQL :5432, Redis :6379, Milvus :19530
```

Backend: ruff (check + format) and pytest (`uv run pytest -q`, 21 tests in `tests/`). Frontend has `npm run test` (vitest, 11 tests) and `npm run build` (vue-tsc + vite).

## 开发环境（WSL）

项目运行在 WSL（Ubuntu-20.04）里，**所有命令都在 WSL 终端内执行**，不要在 Windows 的 cmd/PowerShell 里跑（尤其 npm——Windows 的 npm 无法在 WSL 挂载路径 `\\wsl.localhost\...` 上创建符号链接，会报 `ENOTEMPTY` / `EISDIR`）。

进入 WSL 终端（默认用户的交互式登录 shell）：

```bash
wsl -d Ubuntu-20.04
```

如果从 Windows 侧调用 WSL 里的命令（例如 Claude Code 跑在 Windows 上，用 Bash 工具执行），**必须用 `bash -ic` 交互式 shell**，否则 `.bashrc` / nvm 不会加载：

```bash
# ✅ 正确 —— 加载用户环境，node 为 v26.5.0
wsl -d Ubuntu-20.04 bash -ic "node -v"        # v26.5.0

# ❌ 错误 —— 非交互 shell 解析到系统 Node v10.19.0
wsl -d Ubuntu-20.04 bash -lc "node -v"        # v10.19.0
```

注意：`bash -ic` 会执行整个 `.bashrc`（含 nvm、ROS 等环境变量输出），输出里会有噪音，属正常。

Node / npm 版本：`node v26.5.0`（nvm 管理，位于 `~/.nvm/versions/node/`）、`npm 11.17.0`。

## Architecture (current)

Backend is **restructured** into subdirectories — do not assume flat `backend/*.py`:

```
backend/
  core/app.py          # FastAPI entry, CORS, static mount
  api/api.py           # Routes
  api/schemas.py       # Pydantic models
  agent/agent.py       # Thin orchestration layer driving the unified chat graph; ConversationStorage (append-only + superseded_by), persistent note, session titles
  agent/tools.py       # Tools (weather), emit_rag_step → get_stream_writer bridge
  rag/chat_graph.py    # Unified chat StateGraph: prepare_context → route → {weather, retrieve→check_hitl} → generate; interrupt/Command(resume)
  rag/hitl_detect.py   # HITL detection (clarify / scope_select / no_knowledge) + rag_trace normalization
  rag/rag_pipeline.py  # RAG workflow: retrieve → grade → rewrite → expand → answer
  rag/rag_utils.py     # grade_documents, rewrite_question, auto-merging
  models/models.py     # SQLAlchemy ORM (User, ChatSession, ChatMessage, ParentChunk)
  infra/auth.py        # JWT auth, password hashing (PBKDF2-SHA256), RBAC deps
  infra/database.py    # SQLAlchemy engine, init_db()
  infra/checkpointer.py# PostgresSaver checkpointer (get_checkpointer / init_checkpointer) — HITL state in PostgreSQL
  infra/cache.py       # Redis JSON cache
  document/            # PDF/Word/Excel loader, 3-level sliding window chunking
  vector/              # Embedding (BAAI/bge-m3 dense + BM25 sparse), Milvus read/write
  jobs/                # Background upload/delete jobs
```

Frontend is now **Vite + TypeScript + Vue 3** (not CDN single-file). Source in `frontend/src/`, build output in `frontend/dist/`.

聊天流程是统一图（不是旧的多条 if/else 分支）：`backend/agent/agent.py` 只是薄编排层（加载历史 + 持久化笔记/标题 + 收尾落库），把输入交给 `backend/rag/chat_graph.py` 的 StateGraph 跑一轮 —— `prepare_context → route → {weather | retrieve→check_hitl} → generate`。HITL 在 `check_hitl` 节点用 LangGraph 原生 `interrupt()` 挂起，检查点经 `backend/infra/checkpointer.py` 的 PostgresSaver 落 PostgreSQL。

## Critical gotchas

1. **Entry point changed**: Backend entry is `backend/core/app.py`, NOT `backend/app.py`.
2. **RAG 步骤/内容经 custom 流事件输出**：图节点内用 `get_stream_writer()`（`from langgraph.config import get_stream_writer`）发送 `{"type": "rag_step"|"content", ...}`，编排层转成 SSE；`backend/agent/tools.py` 的 `emit_rag_step` 是对它的封装（tools 里调用，图外调用静默跳过）。旧的 `set_rag_step_queue()` 机制已删除。
3. **服务端 BM25 Function**：`sparse_embedding` 由 Milvus 在插入时根据 `text` 自动生成，客户端不再计算/上传稀疏向量，也不再有 `data/bm25_state.json`。集合 schema 含 BM25 Function（text 需 `enable_analyzer` + `enable_match`）后与旧 schema 不兼容，**升级需 drop collection 并重新上传文档**。
4. **`CLAUDE.md` is tracked in git** (shared).
5. **`data/` is gitignored** — includes uploaded documents.
6. **Frontend build required** before backend serves static files — `cd frontend && npm run build` outputs to `frontend/dist/`.
7. **Docker Compose startup order matters** — Milvus needs etcd + minio healthy first. Use `docker compose up -d` and wait for health checks.
8. **`DENSE_EMBEDDING_DIM` must match Milvus collection schema** — changing the embedding model requires recreating the collection.
9. **本地 Milvus 连接会被 `http(s)_proxy` 劫持**：gRPC 的 `no_proxy` 解析不支持通配符（如 `127.*`），shell 里设了 `http_proxy` 时 `127.0.0.1:19530` 会被误走代理，报 `Fail connecting to server ... illegal connection params or server unavailable`。`milvus_client.py` 在建连前会把 `127.0.0.1`/`localhost` 显式加入 `no_proxy`，无需手动处理。
10. **HITL 用 LangGraph 原生 `interrupt`/`Command(resume)`**：`check_hitl` 节点 `interrupt(hitl_value)` 挂起，用户补充作为下一轮 `Command(resume=user_text)` 恢复。**import 位置（已实测，勿改动）**：`from langgraph.types import interrupt, Command`（`Command` 不在 `langgraph.graph`）；`get_stream_writer` 从 `langgraph.config` import。
11. **Checkpoint 持久化到 PostgreSQL**：图用 `build_chat_graph(checkpointer=get_checkpointer())`；`backend/infra/checkpointer.py` 提供 PostgresSaver 进程级单例（`init_checkpointer()` 启动时建表，DSN 需 psycopg v3 格式、去 SQLAlchemy 方言前缀）。**thread_id = `{user_id}:{session_id}`**（`_session_thread_config`），同一个 thread 才能 resume。
12. **流式路径中断检测用 `graph.get_state(config)`，不是 stream 事件**：`interrupt` 经 `stream`/`astream` 时流会**静默结束**（不产生 `__interrupt__` 事件）；检测 `snap.tasks[0].interrupts[0].value`。resume 时被中断的节点会从头重跑（detect 每轮 HITL 多执行一次重入），检测逻辑须对同输入确定（router temperature=0）。
13. **`LLM_FAST_MODEL`**：持久化笔记/标题等轻任务用 `_get_fast_model()`（缺省回退 `LLM_MODEL`，temperature=0）。
14. **`POST /chat/hitl/cancel`**：放弃当前挂起的 HITL —— `get_checkpointer().delete_thread(thread_id)` 删除该会话 checkpoint 线程（注意 `delete_thread` 签名是 `(thread_id: str)`，传 config dict 会静默失效），下一条消息按新问题处理。

## Key env vars

LLM: `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_FAST_MODEL`（持久化笔记/标题等轻任务模型，缺省回退 `LLM_MODEL`）. Embedding: `EMBEDDING_MODEL`, `DENSE_EMBEDDING_DIM`. Milvus: `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION`. DB/Cache: `DATABASE_URL`, `REDIS_URL`. Auth: `JWT_SECRET_KEY`, `ADMIN_INVITE_CODE`.
