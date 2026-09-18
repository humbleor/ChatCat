"""Blue/green migration utility for rebuilding the knowledge base in a new collection."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from backend.document.document_loader import DocumentLoader
from backend.document.document_registry import ACTIVE, SOFT_DELETED, document_registry
from backend.document.parent_chunk_store import ParentChunkStore
from backend.infra.database import init_db
from backend.vector.milvus_client import MilvusSettings, MilvusStore


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def _build_store(collection_name: str) -> MilvusStore:
    settings = replace(
        MilvusSettings.from_env(),
        collection_name=collection_name,
    )
    return MilvusStore(settings=settings)


def _verify_document(store: MilvusStore, document_id: str, expected: int, status: str) -> None:
    total = store.count(f'document_id == "{document_id}"')
    active = store.count(f'document_id == "{document_id}" and is_deleted == false')
    if total != expected:
        raise RuntimeError(f"Vector count mismatch for {document_id}: expected={expected}, actual={total}")
    if status == SOFT_DELETED and active:
        raise RuntimeError(f"Soft-deleted document {document_id} still has {active} active chunks")
    if status == ACTIVE and active != expected:
        raise RuntimeError(f"Active count mismatch for {document_id}: expected={expected}, actual={active}")


def migrate(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_dir).resolve()
    entries = _load_manifest(Path(args.manifest))
    init_db()

    target_store = _build_store(args.target_collection)
    target_store.init_collection()
    writer = None
    loader = DocumentLoader()
    parent_store = ParentChunkStore()

    report: list[dict] = []
    failures = 0

    for entry in entries:
        filename = str(entry.get("filename") or "").strip()
        requested_status = str(entry.get("status") or ACTIVE).strip()
        if requested_status == "excluded":
            report.append({"filename": filename, "status": "excluded"})
            continue
        if requested_status not in {ACTIVE, SOFT_DELETED}:
            report.append(
                {
                    "filename": filename,
                    "status": "failed",
                    "error": f"Unsupported status: {requested_status}",
                }
            )
            failures += 1
            continue

        file_path = source_dir / filename
        if not file_path.is_file():
            report.append(
                {
                    "filename": filename,
                    "status": "failed",
                    "error": "Source file not found",
                }
            )
            failures += 1
            continue

        document_id = str(entry.get("document_id") or _file_sha256(file_path))
        try:
            existing = target_store.count(f'document_id == "{document_id}"')
            if existing and args.resume:
                registry_item = document_registry.get(document_id)
                expected = int(
                    entry.get("leaf_chunk_count") or (registry_item or {}).get("leaf_chunk_count") or existing
                )
                _verify_document(
                    target_store,
                    document_id,
                    expected,
                    requested_status,
                )
                report.append(
                    {
                        "filename": filename,
                        "document_id": document_id,
                        "status": "verified",
                        "leaf_chunks": expected,
                    }
                )
                continue
            if existing:
                raise RuntimeError("Target already contains this document; use --resume or a fresh collection")

            documents = loader.load_document(
                str(file_path),
                filename,
                document_id=document_id,
                is_deleted=requested_status == SOFT_DELETED,
            )
            parent_docs = [doc for doc in documents if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
            leaf_docs = [doc for doc in documents if int(doc.get("chunk_level", 0) or 0) == 3]
            if not leaf_docs:
                raise RuntimeError("No leaf chunks generated")

            parent_store.upsert_documents(parent_docs)
            if writer is None:
                from backend.vector.embedding import embedding_service
                from backend.vector.milvus_writer import MilvusWriter

                writer = MilvusWriter(
                    embedding_service=embedding_service,
                    milvus_store=target_store,
                )
            writer.write_documents(leaf_docs)
            document_registry.upsert(
                document_id=document_id,
                filename=filename,
                file_path=str(file_path),
                file_type=str(leaf_docs[0].get("file_type", "")),
                status=requested_status,
                leaf_chunk_count=len(leaf_docs),
                parent_chunk_count=len(parent_docs),
            )
            _verify_document(
                target_store,
                document_id,
                len(leaf_docs),
                requested_status,
            )
            report.append(
                {
                    "filename": filename,
                    "document_id": document_id,
                    "status": requested_status,
                    "leaf_chunks": len(leaf_docs),
                    "parent_chunks": len(parent_docs),
                }
            )
        except Exception as exc:
            failures += 1
            report.append(
                {
                    "filename": filename,
                    "document_id": document_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if not args.keep_going:
                break

    output = {
        "target_collection": args.target_collection,
        "documents": report,
        "failed": failures,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-dir", default=str(PROJECT_ROOT / "data" / "documents"))
    parser.add_argument("--target-collection", default="embeddings_collection_v2")
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "data" / "migrations" / "knowledge_v2_report.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(migrate(build_parser().parse_args()))
