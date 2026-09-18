"""PostgreSQL-backed lifecycle registry for knowledge-base documents."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from backend.infra.database import SessionLocal
from backend.models.models import KnowledgeDocument

ACTIVE = "active"
INGESTING = "ingesting"
SOFT_DELETING = "soft_deleting"
SOFT_DELETED = "soft_deleted"
PURGING = "purging"
PURGED = "purged"
FAILED = "failed"


class DocumentRegistry:
    @staticmethod
    def _to_dict(item: KnowledgeDocument) -> dict:
        return {
            "document_id": item.document_id,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "status": item.status,
            "leaf_chunk_count": item.leaf_chunk_count,
            "parent_chunk_count": item.parent_chunk_count,
            "error_message": item.error_message,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "deleted_at": item.deleted_at,
            "purged_at": item.purged_at,
        }

    def create(
        self,
        *,
        filename: str,
        file_path: str,
        file_type: str = "",
        document_id: str | None = None,
        status: str = INGESTING,
    ) -> str:
        document_id = document_id or uuid4().hex
        db = SessionLocal()
        try:
            item = KnowledgeDocument(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                file_path=file_path,
                status=status,
                updated_at=datetime.utcnow(),
            )
            db.add(item)
            db.commit()
            return document_id
        finally:
            db.close()

    def upsert(
        self,
        *,
        document_id: str,
        filename: str,
        file_path: str,
        file_type: str,
        status: str,
        leaf_chunk_count: int = 0,
        parent_chunk_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        db = SessionLocal()
        try:
            item = db.get(KnowledgeDocument, document_id)
            if item is None:
                item = KnowledgeDocument(document_id=document_id, filename=filename)
                db.add(item)
            item.filename = filename
            item.file_path = file_path
            item.file_type = file_type
            item.status = status
            item.leaf_chunk_count = leaf_chunk_count
            item.parent_chunk_count = parent_chunk_count
            item.error_message = error_message
            item.updated_at = datetime.utcnow()
            item.deleted_at = datetime.utcnow() if status == SOFT_DELETED else None
            db.commit()
        finally:
            db.close()

    def get(self, document_id: str) -> dict | None:
        db = SessionLocal()
        try:
            item = db.get(KnowledgeDocument, document_id)
            return self._to_dict(item) if item else None
        finally:
            db.close()

    def find_active_by_filename(self, filename: str) -> dict | None:
        db = SessionLocal()
        try:
            item = (
                db.query(KnowledgeDocument)
                .filter(
                    KnowledgeDocument.filename == filename,
                    KnowledgeDocument.status == ACTIVE,
                )
                .order_by(KnowledgeDocument.created_at.desc())
                .first()
            )
            return self._to_dict(item) if item else None
        finally:
            db.close()

    def list_active(self) -> list[dict]:
        db = SessionLocal()
        try:
            rows = (
                db.query(KnowledgeDocument)
                .filter(KnowledgeDocument.status == ACTIVE)
                .order_by(KnowledgeDocument.created_at.desc())
                .all()
            )
            return [self._to_dict(item) for item in rows]
        finally:
            db.close()

    def list_purge_candidates(self, retention_days: int, limit: int) -> list[dict]:
        cutoff = datetime.utcnow() - timedelta(days=max(0, retention_days))
        db = SessionLocal()
        try:
            rows = (
                db.query(KnowledgeDocument)
                .filter(
                    KnowledgeDocument.status == SOFT_DELETED,
                    KnowledgeDocument.deleted_at.is_not(None),
                    KnowledgeDocument.deleted_at <= cutoff,
                )
                .order_by(KnowledgeDocument.deleted_at.asc())
                .limit(max(1, limit))
                .all()
            )
            return [self._to_dict(item) for item in rows]
        finally:
            db.close()

    def set_status(
        self,
        document_id: str,
        status: str,
        *,
        leaf_chunk_count: int | None = None,
        parent_chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        db = SessionLocal()
        try:
            item = db.get(KnowledgeDocument, document_id)
            if item is None:
                raise KeyError(f"Unknown document_id: {document_id}")
            item.status = status
            item.error_message = error_message
            item.updated_at = datetime.utcnow()
            if leaf_chunk_count is not None:
                item.leaf_chunk_count = leaf_chunk_count
            if parent_chunk_count is not None:
                item.parent_chunk_count = parent_chunk_count
            if status == SOFT_DELETED and item.deleted_at is None:
                item.deleted_at = datetime.utcnow()
            elif status == ACTIVE:
                item.deleted_at = None
                item.purged_at = None
            elif status == PURGED:
                item.purged_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()


document_registry = DocumentRegistry()
