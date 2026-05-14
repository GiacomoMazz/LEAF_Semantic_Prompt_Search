from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, config):
        self.config = config
        self.model = CrossEncoder(config.reranker_model_name)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        
        pairs = []

        for candidate in candidates:
            pairs.append((query, candidate["text"]))

        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["reranker_score_raw"] = float(score)

        candidates = self.normalize_scores(
            candidates,
            raw_key = "reranker_score_raw",
            normalized_key = "reranker_score"
            )

        for candidate in candidates:
            candidate["final_score"] = candidate["reranker_score"]

        return sorted(
            candidates,
            key = lambda result: result["final_score"],
            reverse = True
        )
    
    @staticmethod
    def normalize_scores(
        results: list[dict],
        raw_key: str,
        normalized_key: str
        ) -> list[dict]:

        if not results:
            return results
        
        raw_scores = []

        for result in results:
            raw_scores.append(float(result.get(raw_key, 0.0)))
        
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if max_score == min_score:
            for result in results:
                result[normalized_key] = 1.0
            return results
        
        for result in results:
            raw_score = float(result.get(raw_key, 0.0))
            result[normalized_key] = (raw_score - min_score) / (max_score - min_score)

        return results