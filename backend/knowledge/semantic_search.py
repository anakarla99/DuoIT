"""Semantic search over the local phonetic-rules knowledge base."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chromadb.api.models.Collection import Collection

from backend.knowledge.init_db import initialize_database


class SemanticSearchError(ValueError):
    """Raised when a semantic-search request is invalid."""


@dataclass(frozen=True)
class PhoneticSearchResult:
    """A phonetic rule returned by a semantic search."""

    rule_id: str
    document: str
    metadata: dict[str, str]
    distance: float


def _validate_query(query: str, limit: int) -> None:
    if not query.strip():
        raise SemanticSearchError("The search query cannot be empty.")
    if limit <= 0:
        raise SemanticSearchError("The search limit must be positive.")


def search_phonetic_rules(
    query: str,
    *,
    limit: int = 3,
    database_path: str | Path | None = None,
    collection_name: str | None = None,
    language: str | None = None,
    collection: Collection | None = None,
) -> list[PhoneticSearchResult]:
    """Return the closest phonetic rules for a natural-language query.

    ChromaDB computes embeddings locally using the collection's configured
    embedding function. An optional language filter is applied by metadata.
    """
    _validate_query(query, limit)
    if language is not None and not language.strip():
        raise SemanticSearchError("The language filter cannot be empty.")

    active_collection = collection or initialize_database(
        database_path=database_path,
        collection_name=collection_name,
    )
    if active_collection.count() == 0:
        return []

    query_kwargs: dict[str, object] = {
        "query_texts": [query],
        "n_results": limit,
        "include": ["documents", "metadatas", "distances"],
    }
    if language is not None:
        query_kwargs["where"] = {"language": language.strip()}

    raw_results = active_collection.query(**query_kwargs)
    ids = raw_results["ids"][0]
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    return [
        PhoneticSearchResult(
            rule_id=rule_id,
            document=document or "",
            metadata=metadata or {},
            distance=distance,
        )
        for rule_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        )
    ]
