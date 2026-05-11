'''
baseline_tfidf.py

This file implements the TF-IDF baseline used as a lexical retrieval
method in the project. It reads the processed dataset, computes cosine
similarity between the query and the prompt texts, and returns ranked results.

'''

from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import PROCESSED_DATA_PATH


RESULT_COLUMNS = [
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
def _load_processed_dataset(processed_path: str) -> pd.DataFrame:
    return pd.read_csv(processed_path)


def _pythonify(value: object) -> object:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


# Purpose:
# Build the TF-IDF representation of the processed prompt texts so the
# baseline can compare user queries against all prompts.
#
# Behavior:
# The function loads the processed dataset, initializes the TF-IDF vectorizer,
# and transforms the text_for_embedding column into a sparse TF-IDF matrix.
#
# Output:
# A tuple containing the processed DataFrame, the fitted TF-IDF vectorizer,
# and the TF-IDF matrix of all prompts.

@lru_cache(maxsize=1)
def _build_tfidf_index(processed_path: str) -> tuple[pd.DataFrame, TfidfVectorizer, object]:
    df = _load_processed_dataset(processed_path)

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text_for_embedding"].fillna(""))

    return df, vectorizer, tfidf_matrix


# Purpose:
# Run the TF-IDF baseline search by comparing the input query with the
# processed prompt texts and ranking the most relevant matches.
#
# Behavior:
# The function transforms the query with the fitted TF-IDF vectorizer,
# computes cosine similarity against all prompt vectors, sorts the prompts
# by similarity score, and keeps the top_k best results.
#
# Output:
# A list of dictionaries containing the selected prompt metadata and the
# corresponding similarity_score for each retrieved result.

def baseline_search(
    query: str,
    top_k: int = 10,
    processed_path: str | Path = PROCESSED_DATA_PATH,
) -> list[dict]:
    if top_k <= 0:
        return []

    df, vectorizer, tfidf_matrix = _build_tfidf_index(str(Path(processed_path)))

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).ravel()

    top_indices = scores.argsort()[::-1][:top_k]
    results = []

    for index in top_indices:
        row = df.iloc[int(index)]
        result = {column: _pythonify(row[column]) for column in RESULT_COLUMNS}
        result["similarity_score"] = float(scores[int(index)])
        results.append(result)

    return results


if __name__ == "__main__":
    sample_results = baseline_search(
        "Write a speech to have a salary raise",
        top_k=5,
    )

    for result in sample_results:
        print(result)


# Final considerations:
# This baseline is intentionally simple and fully separate from embeddings
# and the vector database.
# Its only role is to provide a lexical comparison point for semantic search.
