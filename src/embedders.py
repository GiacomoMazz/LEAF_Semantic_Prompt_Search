from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        
        # embedding all the text in batches
        embeddings = self.model.encode(
            texts,
            batch_size = self.config.embedding_batch_size,
            convert_to_numpy = True,
            normalize_embeddings = self.config.normalize_embeddings,
            show_progress_bar = True

        )

        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        
        # embedding a single search query
        embedding = self.model.encode(
            query,
            convert_to_numpy = True,
            normalize_embeddings = self.config.normalize_embeddings,
            show_progress_bar = False
        )

        return embedding.tolist()
    
if __name__ == "__main__":

    from config import get_config

    config = get_config(

        embedding_model_key="minilm",

        retrieval_mode="semantic",

    )

    embedder = SentenceTransformerEmbedder(config)

    texts = [

        "Prompt for debugging Python code",

        "Prompt for writing a marketing email",

    ]

    embeddings = embedder.embed_texts(texts)

    query_embedding = embedder.embed_query("debug a Python error")

    print(f"Number of document embeddings: {len(embeddings)}")

    print(f"Document embedding dimension: {len(embeddings[0])}")

    print(f"Query embedding dimension: {len(query_embedding)}")