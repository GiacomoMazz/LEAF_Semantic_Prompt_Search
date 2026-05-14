# config.py
"""
Central configuration for the semantic prompt search pipeline.

Creates a SearchConfig class to store all of our variables and constants in one place
Easier to change settings later on.
"""

from pathlib import Path
from dataclasses import dataclass, field



## Paths


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "dataset.json"
VECTOR_DB_DIR = PROJECT_ROOT / "chroma"

## Text construction

# template for build_search_text

TEXT_TEMPLATE = """Title: {title}
Category: {category}
Subcategory: {subcategory}
Tags: {tags}
Difficulty: {difficulty}

Prompt: 
{content}
"""

# metadata fields stored for each vector

METADATA_FIELDS = [
    "title",
    "category",
    "subcategory",
    "tags",
    "difficulty",
    "language",
    "target_model",
    "has_placeholders",
    "placeholders",
    "author_reputation",
    "version",
    "fork_count",
    "likes",
    "upvotes",
    "downvotes",
    "views",
    "uses",
    "created_at"
]

## Embedding settings

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}

DEFAULT_EMBEDDING_MODEL_KEY = "minilm"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_NORMALIZE_EMBEDDINGS = True

# Vector store and retrieval modes

RETRIEVAL_MODES = {
    "semantic",
    "keyword",
    "hybrid"
}

DEFAULT_RETRIEVAL_MODE = "semantic"

VECTOR_STORE = "chroma"
DISTANCE_METRIC = "cosine"

KEYWORD_TOP_K = 10
SEMANTIC_TOP_K = 10
MERGED_TOP_K = 20
FINAL_TOP_K = 5

KEYWORD_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.7


def make_collection_name(model_key: str) -> str:
    # Make collection names model-specific so indexes do not get mixed.
    return f"prompts_{model_key}"

## Reranking

RERANKER_MODELS = {
    "msmarco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

DEFAULT_RERANKER_MODEL_KEY = "msmarco_minilm"

## metadata scoring

DEFAULT_SEARCH_SCORE_WEIGHT = 0.85
DEFAULT_METADATA_SCORE_WEIGHT = 0.15

METADATA_SCORE_FIELDS = {
    "upvotes": 0.30,
    "likes": 0.22,
    "uses": 0.22,
    "fork_count": 0.10,
    "author_reputation": 0.10,
    "views": 0.06,
    "downvotes": -0.10
}

# Runtime config

@dataclass
class SearchConfig:
    embedding_model_key: str = DEFAULT_EMBEDDING_MODEL_KEY
    embedding_model_name: str = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL_KEY]
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS
    search_score_weight: float = DEFAULT_SEARCH_SCORE_WEIGHT
    metadata_score_weight: float = DEFAULT_METADATA_SCORE_WEIGHT

    # not completely necessary for strings but safest
    collection_name: str = field(
        default_factory=lambda: make_collection_name(DEFAULT_EMBEDDING_MODEL_KEY)
    )

    vector_store: str = VECTOR_STORE
    vector_db_dir: Path = VECTOR_DB_DIR
    distance_metric: str = DISTANCE_METRIC

    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE

    keyword_top_k: int = KEYWORD_TOP_K
    semantic_top_k: int = SEMANTIC_TOP_K
    merged_top_k: int = MERGED_TOP_K
    final_top_k: int = FINAL_TOP_K

    keyword_weight: int = KEYWORD_WEIGHT
    semantic_weight: int = SEMANTIC_WEIGHT

    use_reranker: bool = False
    reranker_model_key: str = DEFAULT_RERANKER_MODEL_KEY
    reranker_model_name: str = RERANKER_MODELS[DEFAULT_RERANKER_MODEL_KEY]

    use_metadata_scoring: bool = False


def get_config(
    embedding_model_key: str = DEFAULT_EMBEDDING_MODEL_KEY,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    keyword_top_k: int = KEYWORD_TOP_K,
    semantic_top_k: int = SEMANTIC_TOP_K,
    merged_top_k: int = MERGED_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    keyword_weight: float = KEYWORD_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
    search_score_weight: float = DEFAULT_SEARCH_SCORE_WEIGHT,
    metadata_score_weight: float = DEFAULT_METADATA_SCORE_WEIGHT,
    use_reranker: bool = False,
    use_metadata_scoring: bool = False,
) -> SearchConfig:
    
    # Create a SearchConfig for one experiment/run.

    if embedding_model_key not in EMBEDDING_MODELS:
        valid = ", ".join(EMBEDDING_MODELS)
        raise ValueError(f"Unknown embedding model key '{embedding_model_key}'. Valid: {valid}")
    
    if retrieval_mode not in RETRIEVAL_MODES:
        valid = ", ".join(RETRIEVAL_MODES)
        raise ValueError(f"Unknown retrieval mode '{retrieval_mode}'. Valid: {valid}")

    return SearchConfig(
        embedding_model_key = embedding_model_key,
        embedding_model_name = EMBEDDING_MODELS[embedding_model_key],
        collection_name = make_collection_name(embedding_model_key),
        embedding_batch_size = embedding_batch_size,
        normalize_embeddings = normalize_embeddings,
        retrieval_mode = retrieval_mode,

        keyword_top_k = keyword_top_k,
        semantic_top_k = semantic_top_k,
        merged_top_k = merged_top_k,
        final_top_k = final_top_k,

        keyword_weight = keyword_weight,
        semantic_weight = semantic_weight,

        search_score_weight = search_score_weight,
        metadata_score_weight = metadata_score_weight,

        use_reranker = use_reranker,
        use_metadata_scoring = use_metadata_scoring,
    )