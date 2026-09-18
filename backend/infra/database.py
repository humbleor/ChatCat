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
        # create_all() 不会修改既有 chat_messages；Run 表本身由 metadata.create_all 创建。
        conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS run_id VARCHAR(64) NULL"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chat_messages_run_id ON chat_messages (run_id)"))
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'fk_chat_messages_run_id'
                    ) THEN
                        ALTER TABLE chat_messages
                        ADD CONSTRAINT fk_chat_messages_run_id
                        FOREIGN KEY (run_id) REFERENCES chat_runs(id) ON DELETE SET NULL;
                    END IF;
                END $$;
                """
            )
        )
