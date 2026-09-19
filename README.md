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

### 4) Docker 部署（数据库 + 缓存 + 向量库 + Reranker）

当前仓库的 `docker-compose.yml` 同时承载业务依赖、Milvus 依赖和可选的 TEI Reranker：
- 业务依赖：`postgres`、`redis`
- 向量依赖：`etcd`、`minio`、`standalone`、`attu`
- 重排序服务：`reranker`（位于 `reranker` profile，默认不启动）

```bash
# 启动数据库、缓存和向量库
docker compose up -d

# 启动 CPU Reranker（适合兼容性验证）
docker compose --profile reranker up -d reranker

# 或启动 GPU Reranker（RTX 4000 系列等 Ada GPU）
docker compose -f docker-compose.yml -f docker-compose.reranker-gpu.yml --profile reranker up -d reranker

# 查看状态和日志
docker compose ps
docker compose logs -f standalone
docker logs -f chatcat-reranker

# 停止依赖服务
docker compose down
```

其他 GPU 架构可通过 `TEI_RERANKER_GPU_IMAGE` 选择对应的 TEI 镜像。使用本机 TEI 时，在 `.env` 中配置：

```dotenv
RERANK_PROVIDER=tei
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=http://127.0.0.1:8081
RERANK_API_KEY=
RERANK_TIMEOUT_SECONDS=30
```

模型首次启动时会下载到 `volumes/huggingface/`，后续重建或重启容器会复用该缓存；但每次启动仍需将模型加载到内存或显存。不要删除该目录，否则需要重新下载模型。如果模型下载失败，应优先检查 Hugging Face 网络或镜像源，避免配置不完整的 `HF_ENDPOINT`。

启动后确认 Reranker 已就绪：

```bash
curl --noproxy '*' http://127.0.0.1:8081/health
curl --noproxy '*' http://127.0.0.1:8081/info
docker inspect --format '{{.State.Health.Status}}' chatcat-reranker
```

健康状态应为 `healthy`。如果 ChatCat 后端也运行在同一个 Compose 网络中，将 `RERANK_BINDING_HOST` 改为 `http://reranker:80`。修改 Reranker 环境变量后需要重启后端，因为 Provider 配置在进程启动时加载。Reranker 不改变 `bge-m3` 的 1024 维向量，因此切换 `jina`/`tei` 无需重建 Milvus；服务不可用时会保留原始 RRF 顺序，并在 `rag_trace.rerank_error` 中记录错误。

端口说明：
- PostgreSQL：`5432`
- Redis：`6379`
- Milvus：`19530`
- Milvus 健康检查：`9091`
- MinIO API：`9000`
- MinIO Console：`9001`
- Attu：`8080`
- Reranker：`8081`

### 5) 启动应用并访问
在 Milvus 启动后，运行后端应用：

```bash
# Backend
uv run uvicorn backend.core.app:app --host 0.0.0.0 --port 8000 --reload

# Frontend (proxies to backend:8000)
cd frontend
npm run dev  # Vite on :3000
npm run build
```

浏览器访问：
- 前端页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`


## 代码格式化 / Lint

```bash
# 后端
uv run ruff format backend tests
uv run ruff check --fix backend tests

# 前端
cd frontend && npm run format && npm run lint:fix
```
