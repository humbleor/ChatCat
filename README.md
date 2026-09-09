# ChatCat 项目说明

ChatCat 是一个基于 LangChain Agent 的 RAG 聊天机器人，后端 FastAPI + 前端 Vue3 / TypeScript / Vite。核心能力：混合检索（Milvus 稠密向量 + BM25 稀疏向量）、Jina 重排序、三级滑动窗口分块 + 自动合并、SSE 流式输出 + 实时 RAG 步骤可视化、JWT 鉴权 + RBAC 权限（admin/user）、PostgreSQL 持久化、Redis 缓存。

## 本地部署

### 1) 环境准备
- Python `3.12+`
- 包管理建议：`uv`（也支持 `pip`）
- Docker / Docker Compose（用于启动业务依赖）

### 2) 使用 pyproject 安装依赖
在项目根目录执行：

```bash
# 方式 A：推荐（uv）
uv sync
```

```bash
# 方式 B：pip
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 3) 创建 `.env` 文件
在项目根目录新建 `.env`，可直接使用下面模板，填上你的内容：

```bash
cp .env.example .env
```

### 4) Docker 部署（数据库 + 缓存 + 向量库）
当前仓库的 `docker-compose.yml` 同时承载业务依赖与 Milvus 依赖：
- 业务依赖：`postgres`、`redis`
- 向量依赖：`etcd`、`minio`、`standalone`、`attu`

```bash
# 启动依赖服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志（可选）
docker compose logs -f standalone

# 停止依赖服务
docker compose down

# 停止并清除数据
docker compose down -v
```

端口说明：
- PostgreSQL：`5432`
- Redis：`6379`
- Milvus：`19530`
- Milvus 健康检查：`9091`
- MinIO API：`9000`
- MinIO Console：`9001`
- Attu：`8080`

### 5) 启动应用并访问
在 Milvus 启动后，运行后端应用：

```bash
# Backend
uv run python -m backend.core.app
# 或
uv run uvicorn backend.core.app:app --host 0.0.0.0 --port 8000 --reload

# Frontend (proxies to backend:8000)
cd frontend
npm run dev  # Vite on :3000
npm run build
```

浏览器访问：
- 前端页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

## RAG 检索与查询扩展

`backend/rag/rag_pipeline.py` 的检索链是 `retrieve_initial → grade_documents → (generate_answer | rewrite_question → retrieve_expanded) → END`。初始检索不通过时进入扩展检索，由 router LLM 在三种扩展策略里三选一：

| 策略 | 适用场景 | 扩展方式 |
|---|---|---|
| `step_back` | 含具体名称 / 日期 / 代码等细节 | 抽象成"退步问题" + 退步问题答案，拼成 `expanded_query` 再检索 |
| `hyde` | 模糊、定义型、需解释 | 生成"假设性文档"作为检索 query |
| `complex` | 多实体 / 多主题对比 / 列举（如 "A 和 B 区别"、"A、B、C 三者对比"） | **拆分为子问题分别检索**，合并去重 |

`complex` 由 `backend/rag/rag_utils.py::decompose_question` 实现：

1. router LLM 按 prompt 拆出 `List[str]`，上限 `MAX_SUB_QUESTIONS = 5`
2. 拆不出来时（如 LLM 失败、返回原问题、JSON 解析挂）走 **regex 兜底**：以首个 `和/与/及/对比/区别/差异/分别/`、` 为分界切成左右两半
3. 仍失败时回退到 `[原问题]`

扩展检索结果会写进 trace 的 `sub_questions` / `sub_agent_count` / `synthesis_merged_count` 三个字段；前端 `frontend/src/types/chat.ts` 的 `rewrite_method` 联合类型已包含 `'complex'`。

### 推理模型兼容

项目默认 router 是 DeepSeek-R1 / MiniMax-M3 等推理模型，会在 answer 前输出 `<think>...</think>` 推理块。langchain 的 `with_structured_output` 会把这块连同 JSON 一起喂给 Pydantic，导致 `Invalid JSON` 抛 ValidationError（实测：`<think>The user is askin...\n\ncomplex`）。

`backend/rag/rag_utils.py::strip_think` 剥掉 `<think>...</think>` 后再 parse。`rewrite_question_node` 与 `grade_documents_node` 都改用：

```
raw = model.invoke(prompt).content
cleaned = strip_think(raw)
result = SchemaCls.model_validate_json(cleaned)
```

不再依赖 `with_structured_output` wrapper。换非推理模型时 `strip_think` 是 no-op，行为不变。

## 代码格式化 / Lint

前后端已统一配置代码风格工具，规则如下：

| 端 | 格式化 | Lint |
|----|--------|------|
| 后端（Python） | ruff format / black | ruff check |
| 前端（Vue 3 + TS） | Prettier | ESLint |

行宽统一为 `120`。所有命令在项目根目录（后端）或 `frontend/`（前端）下执行。

### 后端（Python）

配置位于根目录 `pyproject.toml`（`[tool.ruff]` / `[tool.black]`）。

```bash
# 格式化（ruff format 与 black 风格一致，二选一）
uv run ruff format backend tests
# 或
uv run black backend tests

# Lint 检查 + 自动修复（import 排序、未用变量等）
uv run ruff check --fix backend tests

# 只检查不修改
uv run ruff check backend tests
uv run ruff format --check backend tests
```

也可以用全项目范围（`frontend/`、`data/` 已在配置中排除，`.venv` 默认跳过）：

```bash
uv run ruff format .
uv run ruff check --fix .
```

### 前端（Vue 3 + TypeScript）

配置位于 `frontend/eslint.config.js` 与 `frontend/.prettierrc`。

```bash
cd frontend

# 格式化（Prettier）
npm run format         # prettier --write .（全量格式化）
npm run format:check   # prettier --check .（仅检查，适合 CI）

# Lint（ESLint）
npm run lint           # eslint .（检查）
npm run lint:fix       # eslint --fix .（检查 + 自动修复）
```

建议的提交前流程：

```bash
# 后端
uv run ruff format backend tests
uv run ruff check --fix backend tests

# 前端
cd frontend && npm run format && npm run lint:fix
```
