"""
Build Chroma vector indexes for semantic prompt search.

This script loads the raw prompt dataset, preprocesses the records, embeds the
processed prompt texts with each configured embedding model, and stores the
resulting embeddings, documents, IDs, and metadata in persistent ChromaDB
collections.
"""

from config import get_config, EMBEDDING_MODELS
from data_loader import load_raw_data
from preprocessing import preprocess_records
from embedders import SentenceTransformerEmbedder
from vectorstores import ChromaVectorStore

def build_index(embedding_model_key: str) -> None:
    config = get_config(
        embedding_model_key = embedding_model_key,
        retrieval_mode = "semantic",
        embedding_batch_size = 32
    )

    print(f"\nBuilding index for: {embedding_model_key}")
    print(f"Collection name: {config.collection_name}")

    raw_records = load_raw_data()
    processed_records = preprocess_records(raw_records)

    ids = []
    for record in processed_records:
        ids.append(record["id"])

    texts = []
    for record in processed_records:
        texts.append(record["text"])

    metadatas = []
    for record in processed_records:
        metadatas.append(record["metadata"])
    
    embedder = SentenceTransformerEmbedder(config)
    embeddings = embedder.embed_texts(texts)

    store = ChromaVectorStore(config)
    store.add_documents(
        ids = ids,
        texts = texts,
        metadatas = metadatas,
        embeddings = embeddings
    )

    print(f"Stored documents: {store.count()}")

def main():
    for embedding_model_key in EMBEDDING_MODELS:
        build_index(embedding_model_key)

if __name__ == "__main__":
    main()