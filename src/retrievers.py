class SearchRetriever:
    def __init__(
            self,
            config,
            embedder = None,
            vector_store = None,
            keyword_retriever = None
        ):

        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever

    def search(self, query: str) -> list[dict]:
        
        if self.config.retrieval_mode == "semantic":
            return self.semantic_search(query)
        
        if self.config.retrieval_mode == "keyword":
            return self.keyword_search(query)
        
        if self.config.retrieval_mode == "hybrid":
            return self.hybrid_search(query)
        
        raise ValueError(f"Unknown retrieval mode: {self.config.retrieval_mode}")
    
    def semantic_search(self, query: str) -> list[dict]:

        if self.embedder is None or self.vector_store is None:
            raise RuntimeError("Semantic search requires embedder and vector_store.")

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding = query_embedding,
            top_k = self.config.semantic_top_k
        )

        for result in results:
            distance = result.get("distance", 1.0)
            semantic_score = 1.0 - float(distance)

            result["semantic_score"] = semantic_score
            result["keyword_score"] = 0.0
            result["hybrid_score"] = semantic_score
            result["final_score"] = semantic_score

        return results
    
    def keyword_search(self, query:str) -> list[dict]:

        if self.keyword_retriever is None:
            raise RuntimeError("Keyword search requires keyword_retriever.")
        
        results = self.keyword_retriever.search(
            query = query,
            top_k = self.config.keyword_top_k
        )

        for result in results:
            keyword_score = float(result.get("keyword_score", 0.0))

            result["semantic_score"] = 0.0
            result["keyword_score"] = keyword_score
            result["hybrid_score"] = keyword_score
            result["final_score"] = keyword_score

        return results
    
    def hybrid_search(self, query: str) -> list[dict]:

        semantic_results = self.semantic_search(query)
        keyword_results = self.keyword_search(query)

        semantic_results = self.normalize_scores(
            results = semantic_results,
            score_key = "semantic_score"
        )

        keyword_results = self.normalize_scores(
            results = keyword_results,
            score_key = "keyword_score"
        )
        
        merged = self.merge_results(
            semantic_results = semantic_results,
            keyword_results = keyword_results
        )

        ranked = sorted(
            merged,
            key = lambda result: result["final_score"],
            reverse = True
        )

        return ranked[: self.config.merged_top_k]
    
    def merge_results(
            self,
            semantic_results: list[dict],
            keyword_results: list[dict]
        ) -> list[dict]:

        merged_by_id = {}

        for result in semantic_results:
            prompt_id = result["id"]
            semantic_score = result.get("semantic_score", 0.0)

            merged_by_id[prompt_id] = {
                **result,
                "semantic_score": semantic_score,
                "keyword_score": 0.0
            }

        for result in keyword_results:
            prompt_id = result["id"]
            keyword_score = result.get("keyword_score", 0.0)

            if prompt_id not in merged_by_id:
                merged_by_id[prompt_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": keyword_score
                }
            else:
                merged_by_id[prompt_id]["keyword_score"] = keyword_score
        
        for result in merged_by_id.values():
            semantic_score = result.get("semantic_score", 0.0)
            keyword_score = result.get("keyword_score", 0.0)

            hybrid_score = (
                self.config.semantic_weight * semantic_score +
                self.config.keyword_weight * keyword_score
            )

            result["hybrid_score"] = hybrid_score
            result["final_score"] = hybrid_score

        return list(merged_by_id.values())
    
    @staticmethod
    def normalize_scores(results: list[dict], score_key: str) -> list[dict]:
        
        if not results:
            return results
        
        max_score = 0.0

        for result in results:
            score = float(result.get(score_key, 0.0))
            if score > max_score:
                max_score = score

        if max_score <= 0:
            return results
        
        for result in results:
            result[score_key] = float(result.get(score_key, 0.0)) / max_score
        
        return results
