# run with streamlit run ~/src/demo_streamlit.py

import time

import streamlit as st

from config import get_config
from pipelines import SearchPipeline


@st.cache_resource
def load_pipeline(
    embedding_model_key: str,
    retrieval_mode: str,
    use_reranker: bool,
    reranker_model_key: str,
    use_metadata_scoring: bool,
    top_k: int,
):
    config = get_config(
        embedding_model_key = embedding_model_key,
        retrieval_mode = retrieval_mode,
        semantic_top_k = 50,
        keyword_top_k = 50,
        merged_top_k = 75,
        final_top_k = top_k,
        use_reranker = use_reranker,
        reranker_model_key = reranker_model_key,
        use_metadata_scoring = use_metadata_scoring,
    )

    return SearchPipeline(config), config


def main():
    st.set_page_config(
        page_title="Prompt Me If You Can",
        layout="wide",
    )

    st.title("Prompt Me If You Can")

    with st.sidebar:
        st.header("Search Settings")

        embedding_model_key = st.selectbox(
            "Embedding model",
            ["bge_small", "minilm"],
        )

        retrieval_mode = st.selectbox(
            "Retrieval mode",
            ["hybrid", "semantic", "keyword"],
        )

        use_reranker = st.checkbox(
            "Use reranker",
            value=True,
        )

        reranker_model_key = st.selectbox(
            "Reranker model",
            ["msmarco", "tinybert"],
        )

        use_metadata_scoring = st.checkbox(
            "Use metadata scoring",
            value=True,
        )

        top_k = st.slider(
            "Final results",
            min_value = 1,
            max_value = 20,
            value = 10,
        )

    query = st.text_input(
        "Enter a natural language query",
        value="help me make my code better",
    )

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        pipeline, config = load_pipeline(
            embedding_model_key = embedding_model_key,
            retrieval_mode = retrieval_mode,
            use_reranker = use_reranker,
            reranker_model_key = reranker_model_key,
            use_metadata_scoring = use_metadata_scoring,
            top_k = top_k,
        )

        start_time = time.perf_counter()
        results = pipeline.search(query)
        latency = time.perf_counter() - start_time

        st.subheader("Search Summary")

        st.write(
            {
                "query": query,
                "embedding_model": config.embedding_model_key,
                "retrieval_mode": config.retrieval_mode,
                "use_reranker": config.use_reranker,
                "reranker_model": config.reranker_model_key if config.use_reranker else None,
                "use_metadata_scoring": config.use_metadata_scoring,
                "latency_seconds": round(latency, 4),
                "results_returned": len(results),
            }
        )

        st.subheader("Results")

        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            content = str(metadata.get("content", ""))

            with st.expander(
                f"{rank}. {metadata.get('title', 'Untitled')} — score {float(result.get('final_score', 0.0)):.4f}",
                expanded = (rank <= 2),
            ):
                st.write(f"**ID:** {result.get('id')}")
                st.write(f"**Category:** {metadata.get('category', '')}")
                st.write(f"**Subcategory:** {metadata.get('subcategory', '')}")
                st.write(f"**Difficulty:** {metadata.get('difficulty', '')}")

                score_cols = st.columns(5)

                score_cols[0].metric(
                    "Final",
                    f"{float(result.get('final_score', 0.0)):.4f}",
                )

                score_cols[1].metric(
                    "Semantic",
                    f"{float(result.get('semantic_score', 0.0)):.4f}",
                )

                score_cols[2].metric(
                    "Keyword",
                    f"{float(result.get('keyword_score', 0.0)):.4f}",
                )

                score_cols[3].metric(
                    "Reranker",
                    f"{float(result.get('reranker_score', 0.0)):.4f}",
                )

                score_cols[4].metric(
                    "Metadata",
                    f"{float(result.get('metadata_score', 0.0)):.4f}",
                )

                st.write("**Prompt content:**")
                st.write(content)


if __name__ == "__main__":
    main()