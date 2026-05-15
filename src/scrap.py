experiments[f"{prefix}_semantic_reranker_msmarco"] = get_config(
    embedding_model_key=embedding_model_key,
    retrieval_mode="semantic",
    semantic_top_k=50,
    final_top_k=10,
    use_reranker=True,
    reranker_model_key="msmarco_minilm",
    use_metadata_scoring=False,
)

experiments[f"{prefix}_semantic_reranker_tinybert"] = get_config(
    embedding_model_key=embedding_model_key,
    retrieval_mode="semantic",
    semantic_top_k=50,
    final_top_k=10,
    use_reranker=True,
    reranker_model_key="msmarco_tinybert",
    use_metadata_scoring=False,
)

experiments[f"{prefix}_semantic_reranker_metadata_msmarco"] = get_config(
    embedding_model_key=embedding_model_key,
    retrieval_mode="semantic",
    semantic_top_k=50,
    final_top_k=10,
    use_reranker=True,
    reranker_model_key="msmarco_minilm",
    use_metadata_scoring=True,
    search_score_weight=0.85,
    metadata_score_weight=0.15,
)

experiments[f"{prefix}_semantic_reranker_metadata_tinybert"] = get_config(
    embedding_model_key=embedding_model_key,
    retrieval_mode="semantic",
    semantic_top_k=50,
    final_top_k=10,
    use_reranker=True,
    reranker_model_key="msmarco_tinybert",
    use_metadata_scoring=True,
    search_score_weight=0.85,
    metadata_score_weight=0.15,
)