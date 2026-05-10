# embeddings.py
#
# Purpose:
# Generate dense sentence embeddings for the processed prompt texts used by
# the semantic retrieval pipeline.
#
# Behavior:
# The module loads a sentence-transformers model once, then encodes a list
# of texts into normalized embedding vectors suitable for cosine retrieval.
#
# Output:
# NumPy array containing one embedding vector for each input text.

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_embeddings(texts) -> np.ndarray:
    model = _load_embedding_model()
    text_list = [str(text) for text in texts]

    embeddings = model.encode(
        text_list,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=False,
    )

    return embeddings


if __name__ == "__main__":
    sample_embeddings = generate_embeddings(
        [
            "Write a speech to have a salary raise",
            "Write a marketing email for a product launch",
        ]
    )
    print(sample_embeddings.shape)


# Final considerations:
# Embedding generation is kept separate from vector storage so the module
# remains easy to test and reuse.
# The vectors are normalized to align with cosine-based semantic search.
