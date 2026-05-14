# build_index.py
"""
Build the Chroma vector index for semantic prompt search.

This script:
1. loads the raw dataset
2. preprocesses each prompt
3. embeds the processed text
4. stores ids, text, metadata, and embeddings in Chroma
"""

import time

from config import get_config
from data_loader import load_raw_data
from preprocessing import preprocess_records
from embedders import SentenceTransformerEmbedder
from vectorstores import ChromaVectorStore


def main():
    config = get_config(
        embedding_model_key="minilm",
        retrieval_mode="semantic",
        embedding_batch_size=32,
    )

    print("Loading raw data...")
    raw_records = load_raw_data()

    print("Preprocessing records...")
    processed_records = preprocess_records(raw_records)

    ids = [record["id"] for record in processed_records]
    texts = [record["text"] for record in processed_records]
    metadatas = [record["metadata"] for record in processed_records]

    print(f"Raw records: {len(raw_records)}")
    print(f"Processed records: {len(processed_records)}")
    print(f"Embedding model: {config.embedding_model_name}")
    print(f"Collection name: {config.collection_name}")
    print(f"Vector DB dir: {config.vector_db_dir}")

    embedder = SentenceTransformerEmbedder(config)

    print("Embedding texts...")
    start_time = time.perf_counter()
    embeddings = embedder.embed_texts(texts)
    elapsed = time.perf_counter() - start_time

    print(f"Embeddings created: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Embedding time: {elapsed:.2f} seconds")

    store = ChromaVectorStore(config)

    print("Adding documents to Chroma...")
    store.add_documents(
        ids=ids,
        texts=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Chroma collection count: {store.count()}")
    print("Index build completed successfully.")


if __name__ == "__main__":
    main()

