'''
vector_store.py

This file builds and manages the ChromaDB vector store used in the
semantic search pipeline. It prepares the prompt metadata, stores the
embeddings together with the corresponding prompt information, and ensures
that the vector collection is available for retrieval.

'''

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


# Purpose:
# Store prompt embeddings and metadata inside the ChromaDB collection
# so they can be used later for semantic retrieval.
#
# Behavior:
# The function creates or reuses the target collection, optionally resets it,
# converts ids, embeddings, metadata, and documents into the format expected
# by ChromaDB, and uploads the data in batches.
#
# Output:
# No direct return value. The function writes the embeddings and metadata
# into the persistent ChromaDB collection on disk.

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


# Purpose:
# Build the vector store starting from the processed dataset by generating
# embeddings and pairing them with the corresponding prompt metadata.
#
# Behavior:
# The function reads the processed CSV file, generates one embedding for each
# text_for_embedding value, prepares the metadata records, and sends everything
# to the ChromaDB collection through build_vector_store(...).
#
# Output:
# The number of prompts inserted into the vector store.

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


# Purpose:
# Check whether the ChromaDB collection is already complete and rebuild it
# if the stored number of prompts does not match the processed dataset.
#
# Behavior:
# The function compares the number of items currently stored in the collection
# with the number of prompts in the processed dataset. If the collection is
# missing items or is empty, it rebuilds the vector store from scratch.
#
# Output:
# The final number of prompts available in the vector store.

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
