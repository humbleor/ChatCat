"""Offline purge for documents whose soft-delete retention period has elapsed."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from backend.document.document_registry import document_registry
from backend.document.document_service import purge_document
from backend.document.parent_chunk_store import ParentChunkStore
from backend.infra.database import init_db
from backend.vector.milvus_client import get_milvus_store


def run(args: argparse.Namespace) -> int:
    init_db()
    store = get_milvus_store()
    parent_store = ParentChunkStore()
    candidates = document_registry.list_purge_candidates(
        retention_days=args.retention_days,
        limit=args.batch_size,
    )
    results = []
    failures = 0
    for item in candidates:
        document_id = item["document_id"]
        try:
            if args.dry_run:
                results.append(
                    {
                        "document_id": document_id,
                        "filename": item["filename"],
                        "status": "would_purge",
                    }
                )
                continue
            result = purge_document(
                document_id,
                registry=document_registry,
                milvus_store=store,
                parent_store=parent_store,
            )
            results.append(
                {
                    "document_id": document_id,
                    "filename": item["filename"],
                    "status": result["status"],
                }
            )
        except Exception as exc:
            failures += 1
            results.append(
                {
                    "document_id": document_id,
                    "filename": item["filename"],
                    "status": "failed",
                    "error": str(exc),
                }
            )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention-days", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
