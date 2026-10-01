"""PostgreSQL-backed lifecycle registry for knowledge-base documents."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import select

from backend.infra.database import SessionLocal
from backend.models.models import DocumentHead, KnowledgeDocument

ACTIVE = "active"
INGESTING = "ingesting"
SOFT_DELETING = "soft_deleting"
STAGING = "staging"
RETIRING = "retiring"
SOFT_DELETED = "soft_deleted"
PURGING = "purging"
PURGED = "purged"
FAILED = "failed"


class DocumentRegistry:
    @staticmethod
    def _to_dict(item: KnowledgeDocument) -> dict:
        return {
            "document_id": item.document_id,
            "version": item.version,
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
        status: str = STAGING,
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
        with SessionLocal() as db:
            head = db.get(DocumentHead, filename)
            item = db.get(KnowledgeDocument, head.current_document_id) if head and head.current_document_id else None
            return self._to_dict(item) if item and item.status == ACTIVE else None

    @staticmethod
    def lock_head(db, filename: str) -> DocumentHead:
        # Insert before row locking serializes publication of the first version too.
        dialect = db.get_bind().dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        else:
            raise RuntimeError("Document publication requires PostgreSQL")
        db.execute(
            insert(DocumentHead)
            .values(
                filename=filename,
                version_counter=0,
                published_version=0,
            )
            .on_conflict_do_nothing(index_elements=["filename"])
        )
        return db.execute(
            select(DocumentHead)
            .where(DocumentHead.filename == filename)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one()

    def reserve_upload(self, db, *, document_id: str, filename: str, file_path: str) -> None:
        head = self.lock_head(db, filename)
        head.version_counter += 1
        db.add(
            KnowledgeDocument(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                status=STAGING,
                version=head.version_counter,
            )
        )

    def has_head(self, filename: str) -> bool:
        with SessionLocal() as db:
            return db.get(DocumentHead, filename) is not None

    def list_active_ids(self) -> list[str]:
        return [item["document_id"] for item in self.list_active()]

    def retrieval_filter(self) -> str:
        """Freeze visibility for this retrieval; preserve unmanaged legacy files."""
        with SessionLocal() as db:
            heads = db.query(DocumentHead).all()
            active_ids = [item.current_document_id for item in heads if item.current_document_id]
            filenames = [item.filename for item in heads if item.published_version > 0 or item.current_document_id]
        parts = []
        if active_ids:
            parts.append(f"document_id in {json.dumps(active_ids, ensure_ascii=False)}")
        legacy = 'document_id == ""'
        if filenames:
            legacy += f" and filename not in {json.dumps(filenames, ensure_ascii=False)}"
        parts.append(f"({legacy})")
        return "(" + " or ".join(parts) + ")"

    def publish(self, document_id: str, *, force: bool = False) -> bool:
        """Commit visibility only after all stores have been verified."""
        with SessionLocal() as db:
            current = db.get(KnowledgeDocument, document_id)
            if current is None:
                raise KeyError(f"Unknown document_id: {document_id}")
            head = self.lock_head(db, current.filename)
            db.refresh(current)
            if head.current_document_id == document_id and current.status == ACTIVE:
                return True
            if current.status not in {STAGING, INGESTING}:
                return False
            if not force and current.version <= head.published_version:
                current.status = RETIRING
                current.updated_at = datetime.utcnow()
                db.commit()
                return False
            previous = db.get(KnowledgeDocument, head.current_document_id) if head.current_document_id else None
            if previous:
                previous.status = RETIRING
                previous.updated_at = datetime.utcnow()
            current.status = ACTIVE
            current.error_message = None
            current.deleted_at = None
            current.updated_at = datetime.utcnow()
            head.current_document_id = document_id
            head.published_version = max(head.published_version, current.version)
            db.commit()
            return True

    def list_retiring(self, filename: str | None = None) -> list[dict]:
        with SessionLocal() as db:
            query = db.query(KnowledgeDocument).filter(KnowledgeDocument.status.in_((RETIRING, SOFT_DELETING)))
            if filename is not None:
                query = query.filter(KnowledgeDocument.filename == filename)
            return [self._to_dict(item) for item in query.all()]

    def list_active(self) -> list[dict]:
        with SessionLocal() as db:
            rows = (
                db.query(KnowledgeDocument)
                .join(DocumentHead, DocumentHead.current_document_id == KnowledgeDocument.document_id)
                .filter(KnowledgeDocument.status == ACTIVE)
                .order_by(KnowledgeDocument.created_at.desc())
                .all()
            )
            return [self._to_dict(item) for item in rows]

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
            if status != ACTIVE:
                head = self.lock_head(db, item.filename)
                db.refresh(item)
                if head.current_document_id == document_id:
                    head.current_document_id = None
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
