import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/chatcat_app",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    # Delayed import to avoid circular dependency.
    from backend.models import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    # create_all() does not alter existing tables. Keep this migration
    # idempotent so existing databases gain the document lifecycle columns.
    with engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE parent_chunks ADD COLUMN IF NOT EXISTS document_id VARCHAR(64) NOT NULL DEFAULT ''")
        )
        conn.execute(
            text("ALTER TABLE parent_chunks ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE")
        )
        conn.execute(text("ALTER TABLE parent_chunks ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP NULL"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_parent_chunks_document_id ON parent_chunks (document_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_parent_chunks_is_deleted ON parent_chunks (is_deleted)"))
