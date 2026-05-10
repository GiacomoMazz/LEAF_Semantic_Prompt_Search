# retrieval.py
#
# Purpose:
# Implement semantic retrieval over the ChromaDB vector store using query
# embeddings and cosine-style similarity scoring.
#
# Behavior:
# The module encodes the user query with the same embedding model used for
# the prompts, queries the Chroma collection, converts returned cosine
# distance values into similarity_score, and returns ranked results.
#
# Output:
# Ranked list of dictionaries containing retrieved prompt metadata and
# similarity_score. The module also exposes a helper to measure retrieval
# latency over a set of queries.

from __future__ import annotations

from time import perf_counter

from embeddings import generate_embeddings
from vector_store import DEFAULT_COLLECTION_NAME, _get_or_create_collection


def retrieve(query: str, top_k: int = 50) -> list[dict]:
    if top_k <= 0:
        return []

    collection = _get_or_create_collection(DEFAULT_COLLECTION_NAME)
    collection_count = collection.count()
    if collection_count == 0:
        raise RuntimeError(
            "The ChromaDB collection is empty. Build the vector store before retrieval."
        )

    effective_top_k = min(top_k, collection_count)
    query_embedding = generate_embeddings([query])[0].tolist()
    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_top_k,
        include=["metadatas", "distances", "documents"],
    )

    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    documents = response.get("documents", [[]])[0]
    ids = response.get("ids", [[]])[0]

    results = []
    for prompt_id, metadata, distance, document in zip(ids, metadatas, distances, documents):
        metadata = metadata or {}
        similarity_score = float(1.0 - distance)

        result = {
            "id": metadata.get("id", prompt_id),
            "title": metadata.get("title", ""),
            "content": metadata.get("content", document or ""),
            "category": metadata.get("category", ""),
            "subcategory": metadata.get("subcategory", ""),
            "tags": metadata.get("tags", ""),
            "likes": metadata.get("likes", 0),
            "upvotes": metadata.get("upvotes", 0),
            "downvotes": metadata.get("downvotes", 0),
            "author_reputation": metadata.get("author_reputation", 0),
            "created_at": metadata.get("created_at", ""),
            "similarity_score": similarity_score,
        }
        results.append(result)

    return results


def measure_retrieval_latency(queries: list[str], top_k: int = 50) -> dict:
    if not queries:
        raise ValueError("The query list must contain at least one query.")

    durations = []
    for query in queries:
        start = perf_counter()
        retrieve(query, top_k=top_k)
        durations.append(perf_counter() - start)

    return {
        "query_count": len(queries),
        "average_latency_seconds": sum(durations) / len(durations),
        "min_latency_seconds": min(durations),
        "max_latency_seconds": max(durations),
    }


if __name__ == "__main__":
    sample_results = retrieve(
        "Write a speech to have a salary raise",
        top_k=5,
    )

    for result in sample_results:
        print(result)


# Final considerations:
# Similarity here compares query meaning against prompt meaning and is used
# only to rank retrieved prompts.
# This module stops at semantic retrieval and does not include reranking or
# metadata-aware score fusion.
