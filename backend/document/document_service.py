"""Document lifecycle operations shared by API and offline jobs."""

from __future__ import annotations

import time
from functools import wraps

from backend.document.document_registry import (
    FAILED,
    PURGED,
    PURGING,
    RETIRING,
    SOFT_DELETED,
    SOFT_DELETING,
    STAGING,
    DocumentRegistry,
)
from backend.document.parent_chunk_store import ParentChunkStore
from backend.jobs.persistent_upload_jobs import exclusive_task
from backend.vector.milvus_client import MilvusStore


def _exclusive_lifecycle(function):
    @wraps(function)
    def call(document_id, **kwargs):
        with exclusive_task("document-lifecycle:" + document_id) as acquired:
            if not acquired:
                raise RuntimeError("Document lifecycle operation is already running")
            return function(document_id, **kwargs)

    return call


@_exclusive_lifecycle
def soft_delete_document(
    document_id: str,
    *,
    registry: DocumentRegistry,
    milvus_store: MilvusStore,
    parent_store: ParentChunkStore,
    retiring_only: bool = False,
) -> dict:
    document = registry.get(document_id)
    if not document:
        raise KeyError(f"Unknown document_id: {document_id}")
    if retiring_only and document["status"] not in {RETIRING, SOFT_DELETING}:
        return document
    if document["status"] in {SOFT_DELETED, PURGING, PURGED}:
        return document

    registry.set_status(document_id, SOFT_DELETING)
    deleted_at = int(time.time())
    try:
        vector_count = milvus_store.set_document_deleted(
            document_id,
            is_deleted=True,
            deleted_at=deleted_at,
        )
        parent_count = parent_store.set_document_deleted(document_id, is_deleted=True)
        remaining = milvus_store.count(f'document_id == "{document_id}" and is_deleted == false')
        if remaining:
            raise RuntimeError(f"Soft delete verification failed: {remaining} active chunks remain")
        registry.set_status(document_id, SOFT_DELETED)
        return {
            **(registry.get(document_id) or document),
            "vector_count": vector_count,
            "parent_count": parent_count,
        }
    except Exception as exc:
        registry.set_status(
            document_id,
            RETIRING if document["status"] in {RETIRING, SOFT_DELETING} else FAILED,
            error_message=str(exc),
        )
        raise


@_exclusive_lifecycle
def restore_document(
    document_id: str,
    *,
    registry: DocumentRegistry,
    milvus_store: MilvusStore,
    parent_store: ParentChunkStore,
) -> dict:
    document = registry.get(document_id)
    if not document:
        raise KeyError(f"Unknown document_id: {document_id}")
    if document["status"] == "active":
        return document
    if document["status"] == PURGED:
        raise RuntimeError("Purged documents must be re-ingested")

    try:
        vector_count = milvus_store.set_document_deleted(document_id, is_deleted=False)
        parent_count = parent_store.set_document_deleted(document_id, is_deleted=False)
        if (
            document["leaf_chunk_count"] <= 0
            or milvus_store.count(f'document_id == "{document_id}" and is_deleted == false')
            != document["leaf_chunk_count"]
        ):
            raise RuntimeError("Restore vector verification failed")
        if parent_store.count_by_document_id(document_id, include_deleted=False) != document["parent_chunk_count"]:
            raise RuntimeError("Restore parent verification failed")
        registry.set_status(document_id, STAGING)
        if not registry.publish(document_id, force=True):
            raise RuntimeError("Document restore publication failed")
        return {
            **(registry.get(document_id) or document),
            "vector_count": vector_count,
            "parent_count": parent_count,
        }
    except Exception as exc:
        registry.set_status(document_id, FAILED, error_message=str(exc))
        raise


@_exclusive_lifecycle
def purge_document(
    document_id: str,
    *,
    registry: DocumentRegistry,
    milvus_store: MilvusStore,
    parent_store: ParentChunkStore,
) -> dict:
    document = registry.get(document_id)
    if not document:
        raise KeyError(f"Unknown document_id: {document_id}")
    if document["status"] == PURGED:
        return document
    if document["status"] != SOFT_DELETED:
        raise RuntimeError(f"Document is not soft-deleted: {document['status']}")

    registry.set_status(document_id, PURGING)
    try:
        result = milvus_store.delete(f'document_id == "{document_id}" and is_deleted == true')
        parent_count = parent_store.purge_by_document_id(document_id)
        remaining = milvus_store.count(f'document_id == "{document_id}"')
        if remaining:
            raise RuntimeError(f"Purge verification failed: {remaining} chunks remain")
        registry.set_status(document_id, PURGED)
        return {
            **(registry.get(document_id) or document),
            "delete_result": result,
            "parent_count": parent_count,
        }
    except Exception as exc:
        # Keep the item retryable by the next offline purge run.
        registry.set_status(document_id, SOFT_DELETED, error_message=str(exc))
        raise
