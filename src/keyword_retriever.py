from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfKeywordRetriever:
    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            lowercase = True,
            stop_words = "english",
            max_features = 50000,
            ngram_range = (1, 2)
        )

        self.records = []
        self.tfidf_matrix = None
    
    def fit(self, records: list[dict]) -> None:

        # fits and stores texts to tfidf_matrix
        texts = []
        self.records = records
        for record in records:
            texts.append(record["text"])
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int) -> list[dict]:

        if self.tfidf_matrix is None:
            raise RuntimeError("Keyword retriever has not been fitted yet.")
        
        query_vector = self.vectorizer.transform([query])

        # score is calcuated based on cosine similarity to query
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            record = self.records[index]

            results.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "keyword_score": float(scores[index])
                }
            )
        return results
