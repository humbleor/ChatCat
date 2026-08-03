# ChatCat 项目说明

ChatCat 是一个基于 LangChain Agent 的 RAG 聊天机器人，后端 FastAPI + 前端 Vite / TypeScript。核心能力：混合检索（Milvus 稠密向量 + BM25 稀疏向量）、Jina 重排序、三级滑动窗口分块 + 自动合并、SSE 流式输出 + 实时 RAG 步骤可视化、JWT 鉴权 + RBAC 权限（admin/user）、PostgreSQL 持久化、Redis 缓存。

## 本地部署

### 1) 环境准备
- Python `3.12+`
- 包管理建议：`uv`（也支持 `pip`）
- Docker / Docker Compose（用于启动 Milvus 依赖）

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
# 启动向量库依赖
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志（可选）
docker compose logs -f standalone

# 停止并清理
docker compose down

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
