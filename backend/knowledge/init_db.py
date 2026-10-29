"""Initialize the local ChromaDB database used by the knowledge service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv

DEFAULT_COLLECTION_NAME = "phonetic_rules"
DEFAULT_DATABASE_PATH = "./chroma_db"


def resolve_database_path(database_path: str | None = None) -> Path:
    """Resolve and validate the local ChromaDB persistence path."""
    configured_path = database_path or os.getenv("CHROMA_DB_PATH", DEFAULT_DATABASE_PATH)
    if not configured_path.strip():
        raise ValueError("The ChromaDB path cannot be empty.")

    path = Path(configured_path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def initialize_database(
    database_path: str | None = None,
    collection_name: str | None = None,
) -> Collection:
    """Create or open the local ChromaDB collection without deleting data."""
    resolved_path = resolve_database_path(database_path)
    resolved_collection_name = (
        collection_name or os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
    ).strip()
    if not resolved_collection_name:
        raise ValueError("The ChromaDB collection name cannot be empty.")

    client = chromadb.PersistentClient(path=str(resolved_path))
    return client.get_or_create_collection(
        name=resolved_collection_name,
        metadata={"description": "Phonetic and linguistic pronunciation rules"},
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for database initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize the local ChromaDB collection for phonetic rules."
    )
    parser.add_argument(
        "--path",
        default=None,
        help=f"Database directory (default: CHROMA_DB_PATH or {DEFAULT_DATABASE_PATH}).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Collection name (default: CHROMA_COLLECTION_NAME or {DEFAULT_COLLECTION_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    """Load configuration, initialize the database, and report its status."""
    load_dotenv()
    args = parse_args()
    collection = initialize_database(args.path, args.collection)
    print(
        f"ChromaDB initialized: collection={collection.name!r}, "
        f"items={collection.count()}"
    )


if __name__ == "__main__":
    main()
