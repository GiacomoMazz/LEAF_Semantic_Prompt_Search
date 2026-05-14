from data_loader import load_raw_data
from preprocessing import preprocess_records
from embedders import SentenceTransformerEmbedder
from vectorstores import ChromaVectorStore
from keyword_retriever import TfidfKeywordRetriever
from retrievers import SearchRetriever
from rerankers import CrossEncoderReranker
from metadata_scoring import MetadataScorer

class SearchPipeline:
    def __init__(self, config):
        self.config = config

        self.embedder = None
        self.vector_store = None
        self.keyword_retriever = None
        self.retriever = None
        self.reranker = None
        self.metadata_scorer = None

        self._build_components()
    
    def _build_components(self) -> None:

        processed_records = None

        if self.config.retrieval_mode in {"keyword", "hybrid"}:
            raw_records = load_raw_data()
            processed_records = preprocess_records(raw_records)
            
            self.keyword_retriever = TfidfKeywordRetriever(self.config)
            self.keyword_retriever.fit(processed_records)
        
        if self.config.retrieval_mode in {"semantic", "hybrid"}:
            self.embedder = SentenceTransformerEmbedder(self.config)
            self.vector_store = ChromaVectorStore(self.config)

            if self.vector_store.count() == 0:
                raise RuntimeError(
                    f"Chroma collection '{self.config.collection_name}' is empty. "
                    "Run build_index.py first."
                )
        
        self.retriever = SearchRetriever(
            config = self.config,
            embedder = self.embedder,
            vector_store = self.vector_store,
            keyword_retriever = self.keyword_retriever
        )

        if self.config.use_reranker:
            self.reranker = CrossEncoderReranker(self.config)
        
        if self.config.use_metadata_scoring:
            self.metadata_scorer = MetadataScorer(self.config)
        
    def search(self, query: str) -> list[dict]:

        candidates = self.retriever.search(query)

        if self.config.use_reranker:
            candidates = self.reranker.rerank(query, candidates)

        if self.config.use_metadata_scoring:
            candidates = self.metadata_scorer.score(candidates)
        
        return candidates[: self.config.final_top_k]