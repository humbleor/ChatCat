"""Document lifecycle operations shared by API and offline jobs."""

from __future__ import annotations

import time

from backend.document.document_registry import (
    ACTIVE,
    FAILED,
    PURGED,
    PURGING,
    SOFT_DELETED,
    SOFT_DELETING,
    DocumentRegistry,
)
from backend.document.parent_chunk_store import ParentChunkStore
from backend.vector.milvus_client import MilvusStore


def soft_delete_document(
    document_id: str,
    *,
    registry: DocumentRegistry,
    milvus_store: MilvusStore,
    parent_store: ParentChunkStore,
) -> dict:
    document = registry.get(document_id)
    if not document:
        raise KeyError(f"Unknown document_id: {document_id}")
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
        registry.set_status(document_id, FAILED, error_message=str(exc))
        raise


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
    if document["status"] == PURGED:
        raise RuntimeError("Purged documents must be re-ingested")

    try:
        vector_count = milvus_store.set_document_deleted(document_id, is_deleted=False)
        parent_count = parent_store.set_document_deleted(document_id, is_deleted=False)
        registry.set_status(document_id, ACTIVE)
        return {
            **(registry.get(document_id) or document),
            "vector_count": vector_count,
            "parent_count": parent_count,
        }
    except Exception as exc:
        registry.set_status(document_id, FAILED, error_message=str(exc))
        raise


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
