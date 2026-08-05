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
