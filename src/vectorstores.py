import chromadb

class ChromaVectorStore:
    def __init__(self, config):
        self.config = config

        self.client = chromadb.PersistentClient(
            path = (config.vector_db_dir)
        )

        self.collection = self.client.get_or_create_collection(
            name = config.collection_name,
            metadata = {"hsnw:space": config.distance_metric}
        )

    # returns number of documents in collection
    def count(self) -> int:
        return self.collection.count()
    
    def add_documents(
            self,
            ids: list[str],
            texts: list[str],
            metadatas: list[dict],
            embeddings: list[list[float]],
            batch_size: int = 1000
        ) -> None:

        total = len(ids)

        # make sure all our data will align properly
        if not (len(ids) == len(texts) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, texts, metadatas, and embeddings must have the same length.")

        # add documents & embeddings to chroma in batches
        for start in range(0, total, batch_size):
            end = start + batch_size 
            self.collection.upsert(
                ids = ids[start:end],
                documents = texts[start:end],
                metadatas = metadatas[start:end],
                embeddings = embeddings[start:end]
        )
    
    def search(
            self,
            query_embedding: list[float],
            top_k: int,
        ) -> list[dict]:

        # search chroma for the nearest documents using
        # same distance metric from config
        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results = top_k,
            include = ["documents", "metadatas", "distances"]
        )

        output = []

        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                }
            )
        
        return output

