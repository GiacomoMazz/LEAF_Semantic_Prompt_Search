# vector_store.py
#
# Purpose:
# Build and populate the persistent ChromaDB collection used for semantic
# retrieval over prompt embeddings.
#
# Behavior:
# The module creates or opens a Chroma collection configured for cosine
# distance, sanitizes metadata into Chroma-compatible Python values, and
# upserts prompt vectors in batches.
#
# Output:
# Persistent ChromaDB collection stored on disk and ready for query-time
# retrieval.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
import pandas as pd

from embeddings import generate_embeddings
from preprocessing import PROCESSED_DATA_PATH


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "data" / "chroma"
DEFAULT_COLLECTION_NAME = "prompts"
METADATA_COLUMNS = [
    "id",
    "title",
    "content",
    "category",
    "subcategory",
    "tags",
    "likes",
    "upvotes",
    "downvotes",
    "author_reputation",
    "created_at",
]


@lru_cache(maxsize=1)
def _get_persistent_client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def _get_or_create_collection(collection_name: str = DEFAULT_COLLECTION_NAME):
    client = _get_persistent_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _sanitize_metadata_value(value: object) -> object:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _build_metadata_records(df: pd.DataFrame) -> list[dict]:
    records = []

    for _, row in df.iterrows():
        metadata = {
            column: _sanitize_metadata_value(row[column]) for column in METADATA_COLUMNS
        }
        records.append(metadata)

    return records


def build_vector_store(
    ids,
    embeddings,
    metadata,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    documents=None,
    reset_collection: bool = False,
) -> None:
    client = _get_persistent_client()

    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = _get_or_create_collection(collection_name)

    id_list = [str(prompt_id) for prompt_id in ids]
    embedding_list = embeddings.tolist()
    metadata_list = [{key: _sanitize_metadata_value(value) for key, value in record.items()} for record in metadata]

    if documents is None:
        document_list = [str(record.get("content", "")) for record in metadata_list]
    else:
        document_list = [str(document) for document in documents]

    batch_size = 1000
    for start in range(0, len(id_list), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=id_list[start:end],
            embeddings=embedding_list[start:end],
            metadatas=metadata_list[start:end],
            documents=document_list[start:end],
        )


def build_vector_store_from_processed_dataset(
    processed_path: str | Path = PROCESSED_DATA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset_collection: bool = True,
) -> int:
    df = pd.read_csv(processed_path)
    embeddings = generate_embeddings(df["text_for_embedding"].fillna(""))
    metadata = _build_metadata_records(df)

    build_vector_store(
        ids=df["id"].astype(str).tolist(),
        embeddings=embeddings,
        metadata=metadata,
        collection_name=collection_name,
        documents=df["content"].astype(str).tolist(),
        reset_collection=reset_collection,
    )

    return len(df)


def ensure_vector_store(
    processed_path: str | Path = PROCESSED_DATA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    collection = _get_or_create_collection(collection_name)
    count = collection.count()
    expected_count = len(pd.read_csv(processed_path, usecols=["id"]))

    if count != expected_count:
        count = build_vector_store_from_processed_dataset(
            processed_path=processed_path,
            collection_name=collection_name,
            reset_collection=True,
        )

    return count


if __name__ == "__main__":
    total_items = ensure_vector_store()
    print(total_items)


# Final considerations:
# This module is responsible only for persistent vector storage.
# Ranking interpretation and query-time similarity handling are left to the
# retrieval layer.
