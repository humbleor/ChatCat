"""Staged, replayable document ingestion; PostgreSQL controls visibility."""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from backend.document.document_registry import ACTIVE, RETIRING, SOFT_DELETING, STAGING
from backend.document.document_service import soft_delete_document

logger = logging.getLogger(__name__)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save_manifest(path, docs):
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(docs, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


async def save_upload_file(file, file_path):
    """Only expose a complete, fsynced source to the queue."""
    temporary = file_path.with_name(file_path.name + ".part")
    try:
        with temporary.open("xb") as stream:
            while chunk := await file.read(1024 * 1024):
                write = asyncio.create_task(asyncio.to_thread(stream.write, chunk))
                try:
                    await asyncio.shield(write)
                except asyncio.CancelledError:
                    await write  # Do not close the file while the writer thread still uses it.
                    raise
            stream.flush()
            await asyncio.to_thread(os.fsync, stream.fileno())
        os.replace(temporary, file_path)
        _sync_directory(file_path.parent)
        _sync_directory(file_path.parent.parent)
    finally:
        temporary.unlink(missing_ok=True)


class DocumentIngestor:
    def __init__(self, *, loader, parent_store, vector_store, writer, registry, jobs):
        self.loader, self.parent_store, self.vector_store = loader, parent_store, vector_store
        self.writer, self.registry, self.jobs = writer, registry, jobs

    def _chunks(self, file_path, filename, document_id):
        manifest = Path(file_path).parent / "chunks.json"
        if manifest.exists():
            docs = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            docs = self.loader.load_document(
                file_path,
                filename,
                document_id=document_id,
                is_deleted=False,
            )
            if not docs:
                raise ValueError("文档未能提取有效内容")
            _save_manifest(manifest, docs)
        ids = [doc.get("chunk_id") for doc in docs]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("分块标识缺失或重复")
        parents = {doc["chunk_id"]: doc for doc in docs if doc.get("chunk_level") in (1, 2)}
        for doc in docs:
            if doc.get("document_id") != document_id or doc.get("chunk_level") not in (1, 2, 3):
                raise ValueError("分块版本或层级不一致")
            parent_id = doc.get("parent_chunk_id")
            if doc["chunk_level"] != 1 and (
                parent_id not in parents or parents[parent_id]["chunk_level"] != doc["chunk_level"] - 1
            ):
                raise ValueError("分块父子关系不完整")
        return list(parents.values()), [doc for doc in docs if doc["chunk_level"] == 3]

    def process(self, job_id, file_path, filename, document_id):
        step = "parse"
        try:
            current = self.registry.get(document_id)
            if not current:
                raise ValueError("入库版本不存在")
            # Publication can commit before progress does. Never rebuild that version.
            if current["status"] not in {ACTIVE, RETIRING, SOFT_DELETING, "soft_deleted"}:
                if current["status"] != STAGING:
                    raise ValueError("入库版本状态不允许构建")
                self.vector_store.init_collection()
                self.jobs.update_step(job_id, step, 5, message="正在解析与分块")
                parents, leaves = self._chunks(file_path, filename, document_id)
                if not leaves:
                    raise ValueError("未生成可检索叶子分块")
                self.jobs.complete_step(job_id, step, f"父块 {len(parents)} 个，叶子块 {len(leaves)} 个")
                step = "parent_store"
                self.jobs.update_step(job_id, step, 10, message="正在幂等写入父块")
                self.parent_store.upsert_documents(parents)
                count = self.parent_store.count_by_document_id(document_id, include_deleted=False)
                if count != len(parents):
                    raise RuntimeError(f"父块数量不一致：{count}/{len(parents)}")
                self.jobs.complete_step(job_id, step, f"父块已入库：{count} 个")
                step = "vector_store"
                self.jobs.update_step(
                    job_id, step, 0, message="正在向量化入库", total_chunks=len(leaves), processed_chunks=0
                )

                def progress(processed, total):
                    self.jobs.update_step(
                        job_id,
                        "vector_store",
                        processed * 100 // max(1, total),
                        message=f"向量化入库：{processed}/{total}",
                        total_chunks=total,
                        processed_chunks=processed,
                    )

                self.writer.write_documents(leaves, progress_callback=progress)
                self.jobs.complete_step(job_id, step, "向量写入完成")
                step = "publish"
                self.jobs.update_step(job_id, step, 10, message="正在校验父块与叶子块")
                persisted = self.parent_store.get_documents_by_ids([p["chunk_id"] for p in parents])
                if {p["chunk_id"] for p in persisted if p.get("document_id") == document_id} != {
                    p["chunk_id"] for p in parents
                }:
                    raise RuntimeError("父块集合校验失败")
                expr = f'document_id == "{document_id}" and is_deleted == false'
                rows = self.vector_store.query_all(filter_expr=expr, output_fields=["chunk_id"])
                if {row["chunk_id"] for row in rows} != {doc["chunk_id"] for doc in leaves} or len(rows) != len(leaves):
                    raise RuntimeError("叶子块集合校验失败")
                self.registry.upsert(
                    document_id=document_id,
                    filename=filename,
                    file_path=file_path,
                    file_type=leaves[0].get("file_type", ""),
                    status=STAGING,
                    leaf_chunk_count=len(leaves),
                    parent_chunk_count=len(parents),
                )
                published = self.registry.publish(document_id)
                self.jobs.complete_step(job_id, step, "新版本已发布" if published else "已有更新版本，跳过发布")
            step = "cleanup"
            # Retiring rows are a durable cleanup queue independent of this job.
            self.cleanup(filename)
            self.jobs.complete_step(job_id, step, "旧版本已下线，后台按保留间隔软删除")
            self.jobs.complete_job(job_id, f"入库完成：{filename}")
        except Exception as exc:
            logger.exception("文档入库失败：%s，阶段：%s", document_id, step)
            self.jobs.retry_job(job_id, step, str(exc), permanent=isinstance(exc, (ValueError, FileNotFoundError)))

    def cleanup(self, filename=None):
        # Allow ongoing readers to finish with their captured visibility snapshot.
        delay = max(0, float(os.getenv("DOCUMENT_RETIRE_GRACE_SECONDS", "60")))
        cutoff = datetime.utcnow() - timedelta(seconds=delay)
        for previous in self.registry.list_retiring(filename):
            if previous["status"] == RETIRING and previous["updated_at"] > cutoff:
                continue
            document_id = previous["document_id"]
            try:
                self.vector_store.init_collection()
                soft_delete_document(
                    document_id,
                    registry=self.registry,
                    milvus_store=self.vector_store,
                    parent_store=self.parent_store,
                    retiring_only=True,
                )
            except Exception:
                logger.exception("旧版本清理失败，下次扫描重试：%s", document_id)
