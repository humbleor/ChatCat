# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ChatCat 是一个基于 LangChain Agent 的 RAG 聊天机器人，后端 FastAPI + 前端 Vue 3 (CDN)。核心能力：混合检索（Milvus 稠密向量 + BM25 稀疏向量）、Jina 重排序、三级滑动窗口分块 + 自动合并、SSE 流式输出 + 实时 RAG 步骤可视化、JWT 鉴权 + RBAC 权限（admin/user）、PostgreSQL 持久化、Redis 缓存。

## 常用命令

```bash
# 安装依赖
uv sync

# 启动后端
uv run python backend/app.py
# 或
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 启动基础设施（PostgreSQL、Redis、Milvus）
docker compose up -d
```

## 架构

### 后端（`backend/`）

各模块直接互相导入（无 `__init__`），入口文件为 `app.py`。

| 文件 | 职责 |
|------|------|
| `app.py` | FastAPI 应用入口、CORS、前端静态文件挂载 |
| `api.py` | 路由处理：认证、聊天（流式）、会话管理、文档管理 |
| `auth.py` | JWT 鉴权、密码哈希（PBKDF2-SHA256）、角色权限依赖 |
| `models.py` | SQLAlchemy ORM 模型：User、ChatSession、ChatMessage、ParentChunk |
| `database.py` | SQLAlchemy 引擎、会话工厂、`init_db()` 建表入口 |
| `cache.py` | Redis JSON 缓存封装 |
| `agent.py` | LangChain/LangGraph Agent、会话消息加载、摘要压缩 |
| `tools.py` | LangChain 工具：天气查询、知识库检索。**关键**：`emit_rag_step()` 通过 `call_soon_threadsafe` 将 RAG 步骤从线程池推送到异步队列。`set_rag_step_queue()` 必须在主线程调用以捕获事件循环。 |
| `rag_pipeline.py` | RAG 工作流：初始检索 → 评分 → 重写路由 → 扩展检索 → 回答。每个阶段调用 `emit_rag_step()` 推送前端可视化。 |
| `rag_utils.py` | 检索辅助函数（grade_documents、rewrite_question 等） |
| `embedding.py` | HuggingFace 稠密向量（默认 `BAAI/bge-m3`）+ BM25 稀疏向量。全局共享 `embedding_service` 单例。 |
| `document_loader.py` | PDF/Word/Excel 加载 + 三级滑动窗口分块（L1/L2/L3） |
| `parent_chunk_store.py` | 父级分块存储（PostgreSQL + Redis），用于自动合并回取 |
| `milvus_writer.py` | 稠密+稀疏向量写入 Milvus，增量更新 BM25 统计 |
| `milvus_client.py` | Milvus 集合定义、混合检索（RRF）、`query_all` 分页查询 |
| `schemas.py` | Pydantic 请求/响应模型 |
| `upload_jobs.py` | 后台文档上传任务处理 |

### 前端（`frontend/`）

Vue 3 CDN 引入，`marked`（Markdown 渲染）+ `highlight.js`（代码高亮）。单页应用由 `index.html` + `script.js` + `style.css` 组成。通过 `ReadableStream` 解析 SSE 事件，维护思考 → 流式输出的状态机。

### 数据（`data/`）

- `bm25_state.json` — BM25 词表、文档频次、总文档数。上传/删除时增量更新。不被 git 跟踪。
- `documents/` — 上传的原始文件。

## 核心数据流

### 聊天（流式）

```
POST /chat/stream → StreamingResponse (SSE)
  ├── agent.astream(stream_mode="messages") 运行在 asyncio.create_task 中
  │     └── AIMessageChunk 文本 → {"type": "content"} → output_queue
  ├── RAG 工具在线程池中执行
  │     └── emit_rag_step() → loop.call_soon_threadsafe → output_queue
  └── 主循环：await output_queue.get() → yield SSE
```

SSE 事件类型：`content`（文本 token）、`rag_step`（思考步骤）、`trace`（完整 RAG 追踪）、`error`、`[DONE]`。

### 文档上传流程

1. 上传 → 检查同名文件（先移除旧 BM25 统计 + Milvus chunk）
2. 三级滑动窗口分块（L1/L2/L3）
3. L1/L2 父分块 → PostgreSQL + Redis（`parent_chunk_store.py`）
4. L3 叶子分块 → BM25 increment_add → 生成稠密+稀疏向量 → 写入 Milvus
5. BM25 状态持久化到 `data/bm25_state.json`

### RAG 检索流程

1. **初始检索**：Milvus 混合（稠密 + 稀疏 + RRF）→ 重排序 → 自动合并 L3→L2→L1
2. **文档评分**：结构化输出 yes/no — yes 直接回答，no 进入重写
3. **查询重写**：在 step-back / HyDE / complex 策略间路由选择
4. **扩展检索**：用重写后的查询再次检索，去重
5. **生成回答**：Agent 结合上下文流式输出

## 鉴权与权限

- `require_auth` 依赖从 JWT Bearer Token 中解析当前用户。
- `require_admin` 依赖检查用户角色。
- 管理员：文档上传/删除/列表。普通用户：聊天、管理自己的会话。

## 环境变量

参考 `.env.example` 或 README.md。关键变量：
- `LLM_API_KEY`、`LLMMODEL`、`LLM_BASE_URL` — LLM 服务
- `EMBEDDING_MODEL`、`DENSE_EMBEDDING_DIM` — 向量嵌入配置
- `MILVUS_HOST`、`MILVUS_PORT` — 向量数据库
- `DATABASE_URL`、`REDIS_URL` — 持久化与缓存
- `JWT_SECRET_KEY`、`ADMIN_INVITE_CODE` — 认证
