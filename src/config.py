# config.py
"""
Central configuration for the semantic prompt search pipeline.

Creates a SearchConfig class to store all of our variables and constants in one place
Easier to change settings later on.
"""

from pathlib import Path
from dataclasses import dataclass, field



# Paths


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "dataset.json"
VECTOR_DB_DIR = DATA_DIR / "chroma"

# Text construction

TEXT_TEMPLATE = """Title: {title}
Category: {category}
Subcategory: {subcategory}
Tags: {tags}
Prompt: {content}
"""

# Embedding models

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}

DEFAULT_EMBEDDING_MODEL_KEY = "minilm"

# Vector store / retrieval

VECTOR_STORE = "chroma"
DISTANCE_METRIC = "cosine"

RETRIEVAL_TOP_K = 10
FINAL_TOP_K = 5


def make_collection_name(model_key: str) -> str:
    # Make collection names model-specific so indexes do not get mixed.
    return f"prompts_{model_key}"

# Reranking

RERANKER_MODELS = {
    "msmarco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

DEFAULT_RERANKER_MODEL_KEY = "msmarco_minilm"

# Runtime config

@dataclass
class SearchConfig:
    embedding_model_key: str = DEFAULT_EMBEDDING_MODEL_KEY
    embedding_model_name: str = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL_KEY]

    # not completely necessary for strings but safest
    collection_name: str = field(
        default_factory=lambda: make_collection_name(DEFAULT_EMBEDDING_MODEL_KEY)
    )

    vector_store: str = VECTOR_STORE
    vector_db_dir: Path = VECTOR_DB_DIR
    distance_metric: str = DISTANCE_METRIC

    retrieval_top_k: int = RETRIEVAL_TOP_K
    final_top_k: int = FINAL_TOP_K

    use_reranker: bool = False
    reranker_model_key: str = DEFAULT_RERANKER_MODEL_KEY
    reranker_model_name: str = RERANKER_MODELS[DEFAULT_RERANKER_MODEL_KEY]

    use_metadata_scoring: bool = False


def get_config(
    embedding_model_key: str = DEFAULT_EMBEDDING_MODEL_KEY,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    use_reranker: bool = False,
    use_metadata_scoring: bool = False,
) -> SearchConfig:
    
    # Create a SearchConfig for one experiment/run.

    if embedding_model_key not in EMBEDDING_MODELS:
        valid = ", ".join(EMBEDDING_MODELS)
        raise ValueError(f"Unknown embedding model key '{embedding_model_key}'. Valid: {valid}")

    return SearchConfig(
        embedding_model_key = embedding_model_key,
        embedding_model_name = EMBEDDING_MODELS[embedding_model_key],
        collection_name = make_collection_name(embedding_model_key),
        retrieval_top_k = retrieval_top_k,
        final_top_k = final_top_k,
        use_reranker = use_reranker,
        use_metadata_scoring = use_metadata_scoring,
    )