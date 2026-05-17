import math

class MetadataScorer:
    def __init__(self, config):
        self.config = config

    def score(self, results: list[dict]) -> list[dict]:

        if not results:
            return results
        
        for result in results:
            result["search_score_raw"] = float(result.get("final_score", 0.0))

        self._add_raw_metadata_scores(results)

        self._normalize_scores(
            results = results,
            raw_key = "search_score_raw",
            normalized_key = "search_score"
        )

        self._normalize_scores(
            results = results,
            raw_key = "metadata_score_raw",
            normalized_key = "metadata_score"
        )

        for result in results:
            result["final_score"] = (
                self.config.search_score_weight * result["search_score"] +
                self.config.metadata_score_weight * result["metadata_score"]
            )

        return sorted(
            results,
            key = lambda result: result["final_score"],
            reverse = True
        )
    
    def _add_raw_metadata_scores(self, results: list[dict]) -> None:

        for result in results:
            metadata = result.get("metadata", {})
            metadata_score_raw = 0.0

            for field, weight in self.config.metadata_score_fields.items():
                value = self._safe_float(metadata.get(field, 0.0))

                scaled_value = math.log1p(max(value, 0.0))

                metadata_score_raw += weight * scaled_value
            
            result["metadata_score_raw"] = metadata_score_raw

    @staticmethod
    def _safe_float(value) -> float:

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
        
    @staticmethod
    def _normalize_scores(
        results: list[dict],
        raw_key: str,
        normalized_key: str
        ) -> None:

        values = []

        for result in results:
            values.append(result.get(raw_key, 0.0))

        min_value = min(values)
        max_value = max(values)

        if max_value == min_value:
            for result in results:
                result[normalized_key] = 0.0
            return
        
        for result in results:
            raw_value = float(result.get(raw_key, 0.0))
            result[normalized_key] = (raw_value - min_value) / (max_value - min_value)
