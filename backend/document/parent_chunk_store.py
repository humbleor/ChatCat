"""父级分块文档存储（用于 Auto-merging Retriever）"""

from datetime import datetime
from typing import List

from backend.infra.cache import cache
from backend.infra.database import SessionLocal
from backend.models.models import ParentChunk


class ParentChunkStore:
    """基于 PostgreSQL + Redis 的父级分块存储。"""

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "document_id": item.document_id,
            "is_deleted": item.is_deleted,
            "deleted_at": item.deleted_at.isoformat() if item.deleted_at else None,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    def upsert_documents(self, docs: List[dict]) -> int:
        """写入/更新父级分块，返回写入条数。"""
        if not docs:
            return 0

        db = SessionLocal()
        upserted = 0
        try:
            for doc in docs:
                chunk_id = (doc.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue

                record = db.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()
                payload = {
                    "document_id": doc.get("document_id", ""),
                    "is_deleted": bool(doc.get("is_deleted", False)),
                    "deleted_at": None,
                    "text": doc.get("text", ""),
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": int(doc.get("page_number", 0) or 0),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": int(doc.get("chunk_level", 0) or 0),
                    "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
                    "updated_at": datetime.utcnow(),
                }
                cache_payload = {
                    "chunk_id": chunk_id,
                    "text": payload["text"],
                    "document_id": payload["document_id"],
                    "is_deleted": payload["is_deleted"],
                    "deleted_at": None,
                    "filename": payload["filename"],
                    "file_type": payload["file_type"],
                    "file_path": payload["file_path"],
                    "page_number": payload["page_number"],
                    "parent_chunk_id": payload["parent_chunk_id"],
                    "root_chunk_id": payload["root_chunk_id"],
                    "chunk_level": payload["chunk_level"],
                    "chunk_idx": payload["chunk_idx"],
                }
                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)
                else:
                    db.add(ParentChunk(chunk_id=chunk_id, **payload))

                cache.set_json(self._cache_key(chunk_id), cache_payload)
                upserted += 1

            db.commit()
        finally:
            db.close()

        return upserted

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []

        ordered_results = {}
        missing_ids = []
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if not key:
                continue
            cached = cache.get_json(self._cache_key(key))
            if cached and not cached.get("is_deleted", False):
                ordered_results[key] = cached
            else:
                missing_ids.append(key)

        if missing_ids:
            db = SessionLocal()
            try:
                rows = (
                    db.query(ParentChunk)
                    .filter(ParentChunk.chunk_id.in_(missing_ids), ParentChunk.is_deleted.is_(False))
                    .all()
                )
                for row in rows:
                    payload = self._to_dict(row)
                    ordered_results[row.chunk_id] = payload
                    cache.set_json(self._cache_key(row.chunk_id), payload)
            finally:
                db.close()

        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    def set_document_deleted(self, document_id: str, *, is_deleted: bool) -> int:
        if not document_id:
            return 0

        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.document_id == document_id).all()
            now = datetime.utcnow() if is_deleted else None
            for row in rows:
                row.is_deleted = is_deleted
                row.deleted_at = now
                row.updated_at = datetime.utcnow()
                cache.delete(self._cache_key(row.chunk_id))
            db.commit()
            return len(rows)
        finally:
            db.close()

    def count_by_document_id(self, document_id: str, *, include_deleted: bool = True) -> int:
        if not document_id:
            return 0
        db = SessionLocal()
        try:
            query = db.query(ParentChunk).filter(ParentChunk.document_id == document_id)
            if not include_deleted:
                query = query.filter(ParentChunk.is_deleted.is_(False))
            return query.count()
        finally:
            db.close()

    def purge_by_document_id(self, document_id: str) -> int:
        if not document_id:
            return 0
        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.document_id == document_id).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted:
                (db.query(ParentChunk).filter(ParentChunk.document_id == document_id).delete(synchronize_session=False))
                db.commit()
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))
            return deleted
        finally:
            db.close()

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除父级分块，返回删除条数。"""
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))
            return deleted
        finally:
            db.close()
