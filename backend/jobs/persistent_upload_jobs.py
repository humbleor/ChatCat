"""PostgreSQL-backed durable upload queue and progress."""

import hashlib
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

from sqlalchemy import text

from backend.document.document_registry import FAILED, document_registry
from backend.infra.database import SessionLocal
from backend.models.models import DocumentJob

logger = logging.getLogger(__name__)
UPLOAD_STEPS = [
    ("upload", "文档上传"),
    ("parse", "解析与分块"),
    ("parent_store", "父级分块入库"),
    ("vector_store", "向量化入库"),
    ("publish", "校验与发布"),
    ("cleanup", "旧版本下线"),
]
_local_guard = Lock()
_local_locks: dict[str, Lock] = {}


@contextmanager
def exclusive_task(key: str):
    """A session advisory lock survives commits and is released on process death."""
    with SessionLocal() as db:
        bind = db.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite is only used by isolated tests.
        with _local_guard:
            lock = _local_locks.setdefault(key, Lock())
        acquired = lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()
        return
    lock_id = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)
    with bind.connect() as conn:
        acquired = conn.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": lock_id}).scalar()
        conn.commit()
        try:
            yield bool(acquired)
        finally:
            if acquired:
                try:
                    conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock_id})
                    conn.commit()
                except Exception:
                    conn.invalidate()


class PersistentUploadJobManager:
    @staticmethod
    def _dict(row):
        return {
            "job_id": row.job_id,
            "filename": row.filename,
            "status": "running" if row.status == "uploading" else row.status,
            "current_step": row.current_step,
            "message": row.message,
            "total_chunks": row.total_chunks,
            "processed_chunks": row.processed_chunks,
            "error": row.error,
            "created_at": row.created_at.isoformat(),
            "updated_at": row.updated_at.isoformat(),
            "steps": deepcopy(row.steps or []),
            "document_id": row.document_id,
            "file_path": row.file_path,
            "attempts": row.attempts,
        }

    def create_job(self, filename, *, document_id, file_path):
        with SessionLocal() as db:
            # Reserve a monotonic version and its queue entry in one transaction.
            document_registry.reserve_upload(
                db,
                document_id=document_id,
                filename=filename,
                file_path=file_path,
            )
            row = DocumentJob(
                job_id=uuid4().hex,
                job_type="upload",
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                status="pending",
                current_step="upload",
                completion_step="cleanup",
                message="文件已落盘，等待入库",
                steps=[
                    {
                        "key": k,
                        "label": v,
                        "percent": 100 if k == "upload" else 0,
                        "status": "completed" if k == "upload" else "pending",
                        "message": "",
                    }
                    for k, v in UPLOAD_STEPS
                ],
            )
            db.add(row)
            db.commit()
            return self._dict(row)

    def get_job(self, job_id):
        with SessionLocal() as db:
            row = db.get(DocumentJob, job_id)
            return self._dict(row) if row and row.job_type == "upload" else None

    def update_step(
        self, job_id, step_key, percent, status="running", message="", *, total_chunks=None, processed_chunks=None
    ):
        with SessionLocal() as db:
            row = db.get(DocumentJob, job_id)
            if not row:
                return None
            steps = deepcopy(row.steps or [])
            step = next((x for x in steps if x["key"] == step_key), None)
            if not step:
                return None
            step.update(percent=max(0, min(100, int(percent))), status=status, message=message)
            row.steps = steps
            if status == "failed":
                row.status = "failed"
            elif row.status == "uploading":
                if status == "completed" and step_key == "upload":
                    row.status = "pending"
            else:
                row.status = "running"
            row.current_step, row.message, row.updated_at = step_key, message, datetime.utcnow()
            if total_chunks is not None:
                row.total_chunks = int(total_chunks)
            if processed_chunks is not None:
                row.processed_chunks = int(processed_chunks)
            db.commit()
            return self._dict(row)

    def complete_step(self, job_id, step_key, message=""):
        return self.update_step(job_id, step_key, 100, "completed", message)

    def complete_job(self, job_id, message="文档入库完成"):
        with SessionLocal() as db:
            row = db.get(DocumentJob, job_id)
            if not row:
                return None
            steps = deepcopy(row.steps or [])
            for step in steps:
                step.update(percent=100, status="completed")
            row.steps, row.status, row.current_step = steps, "completed", row.completion_step
            row.message, row.error, row.updated_at = message, None, datetime.utcnow()
            db.commit()
            return self._dict(row)

    def fail_job(self, job_id, step_key, error):
        self.update_step(job_id, step_key, 0, "failed", error)
        with SessionLocal() as db:
            row = db.get(DocumentJob, job_id)
            if row:
                row.error = error
                db.commit()
                return self._dict(row)
        return None

    def retry_job(self, job_id, step_key, error, *, permanent=False):
        job = self.get_job(job_id)
        current = document_registry.get(job["document_id"]) if job else None
        with SessionLocal() as db:
            row = db.get(DocumentJob, job_id)
            if not row:
                return
            exhausted = permanent or row.attempts >= max(1, int(os.getenv("UPLOAD_MAX_ATTEMPTS", "3")))
            if current and current["status"] in {"active", "retiring", "soft_deleting", "soft_deleted"}:
                exhausted = False  # Publication already committed; only completion bookkeeping remains.
            row.status = "failed" if exhausted else "pending"
            row.error = str(error)
            row.message = str(error) if exhausted else f"处理失败，稍后重试：{error}"
            row.current_step, row.updated_at = step_key, datetime.utcnow()
            steps = deepcopy(row.steps or [])
            for step in steps:
                if step["key"] == step_key:
                    step.update(status="failed" if exhausted else "pending", message=row.message)
            row.steps = steps
            db.commit()
            document_id = row.document_id
        current = document_registry.get(document_id)
        if exhausted and current and current["status"] == "staging":
            document_registry.set_status(document_id, FAILED, error_message=str(error))

    def list_jobs(self):
        with SessionLocal() as db:
            rows = (
                db.query(DocumentJob)
                .filter(DocumentJob.job_type == "upload")
                .order_by(DocumentJob.created_at.desc())
                .all()
            )
            return [self._dict(row) for row in rows]

    def list_runnable(self):
        with SessionLocal() as db:
            rows = (
                db.query(DocumentJob)
                .filter(
                    DocumentJob.job_type == "upload",
                    DocumentJob.status.in_(("uploading", "pending", "running")),
                )
                .order_by(DocumentJob.created_at)
                .all()
            )
            delay = max(0, float(os.getenv("UPLOAD_RETRY_DELAY_SECONDS", "10")))
            cutoff = datetime.utcnow() - timedelta(seconds=delay)
            return [
                self._dict(row)
                for row in rows
                if (row.status == "running" or not row.attempts or row.updated_at <= cutoff)
                and (row.status != "uploading" or Path(row.file_path).is_file())
            ]

    def claim(self, job_id):
        with SessionLocal() as db:
            row = db.query(DocumentJob).filter(DocumentJob.job_id == job_id).with_for_update().first()
            if not row or row.status not in {"pending", "running", "uploading"}:
                return None
            if row.status == "uploading" and not Path(row.file_path).is_file():
                return None
            row.status, row.attempts, row.updated_at = "running", row.attempts + 1, datetime.utcnow()
            db.commit()
            return self._dict(row)


class UploadWorker:
    """A bounded in-process consumer; PostgreSQL is the durable queue."""

    def __init__(self, process, cleanup):
        self.process, self.cleanup = process, cleanup
        self.stop_event = Event()
        self.thread = Thread(target=self._run, name="document-upload-worker", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                for candidate in upload_job_manager.list_runnable():
                    if self.stop_event.is_set():
                        break
                    with exclusive_task("upload:" + candidate["job_id"]) as acquired:
                        if acquired:
                            job = upload_job_manager.claim(candidate["job_id"])
                            if job:
                                self.process(job["job_id"], job["file_path"], job["filename"], job["document_id"])
                if not self.stop_event.is_set():
                    with exclusive_task("document-retirement") as acquired:
                        if acquired:
                            self.cleanup()
            except Exception:
                logger.exception("文档后台队列执行失败，将继续重试")
            self.stop_event.wait(2)


upload_job_manager = PersistentUploadJobManager()
