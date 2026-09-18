import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend" / "dist"
load_dotenv(BASE_DIR / ".env")
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    from backend.api import api as api_module
    from backend.infra.checkpointer import init_checkpointer
    from backend.infra.database import init_db
    from backend.vector.embedding import embedding_service

    preload_mode = os.getenv("EMBEDDING_PRELOAD", "blocking").strip().lower()
    if preload_mode not in {"blocking", "background", "disabled"}:
        raise ValueError("EMBEDDING_PRELOAD 必须是 blocking、background 或 disabled")

    async def _background_warmup() -> None:
        try:
            await asyncio.to_thread(embedding_service.warmup)
        except Exception:
            logger.exception("后台预热嵌入模型失败")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.core_ready = False
        init_db()
        init_checkpointer()
        if preload_mode == "blocking":
            await asyncio.to_thread(embedding_service.warmup)
        elif preload_mode == "background":
            app.state.embedding_warmup_task = asyncio.create_task(_background_warmup())
        app.state.core_ready = True
        yield
        task = getattr(app.state, "embedding_warmup_task", None)
        if task is not None and not task.done():
            task.cancel()

    app = FastAPI(title="ChatCat Bot API", lifespan=lifespan)

    @app.get("/livez", include_in_schema=False)
    async def livez():
        return {"status": "alive"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(response: Response):
        embedding_status = embedding_service.status
        embedding_ready = preload_mode == "disabled" or embedding_status["state"] == "ready"
        ready = bool(getattr(app.state, "core_ready", False) and embedding_ready)
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "ready" if ready else "not_ready",
            "embedding_preload": preload_mode,
            "embedding": embedding_status,
        }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(api_module.router)

    # serve frontend static files at root
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
