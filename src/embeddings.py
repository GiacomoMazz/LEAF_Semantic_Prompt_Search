'''
embeddings.py

This file generates dense vector representations of the prompt texts
using a sentence-transformers model. It loads the embedding model,
encodes the input texts, and returns normalized embeddings that will
later be stored in the vector database and used for semantic retrieval.

'''

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_SMALL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# refactor - changed the load helper to take any model as an input so we can
# test other embedding models. We will want to check if bge is a better technique

@lru_cache(maxsize=1)
def _load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    return SentenceTransformer(model_name)

'''
Purpose:
Convert the input texts into dense semantic embeddings that can be
compared through cosine similarity in the retrieval pipeline.

Behavior:
The function loads the sentence-transformers model, converts each input
text to string format, and encodes the texts into normalized embedding vectors.

Output:
A NumPy array containing one embedding vector for each input text.
'''

# refactor - added model_name to parameters to allow for different embedding techniques

def generate_embeddings(texts, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    model = _load_embedding_model(model_name)
    text_list = [str(text) for text in texts]

    embeddings = model.encode(
        text_list,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=False,
    )

    return embeddings

# refactor - updated test here to also test the bge model as well

if __name__ == "__main__":
    sample_texts = [
    "Write a speech to have a salary raise",
    "Write a marketing email for a product launch"
    ]

    sample_embeddings = generate_embeddings(sample_texts)
    bge_embeddings = generate_embeddings(sample_texts, BGE_SMALL_EMBEDDING_MODEL)

    print(sample_embeddings.shape)
    print(bge_embeddings.shape)

'''
Final considerations:

Embedding generation is kept separate from vector storage so the module
remains easy to test and reuse.
The vectors are normalized to align with cosine-based semantic search.
'''