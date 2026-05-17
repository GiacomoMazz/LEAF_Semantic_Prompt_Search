# LEAF Semantic Prompt Search

This project implements a configurable semantic prompt search system for a dataset of 20,000 prompts provided for the LEAF PromptKaban project.

The system supports keyword search, semantic vector search, hybrid retrieval, cross-encoder reranking, and metadata-aware scoring. It also includes an evaluation workflow based on manually labeled relevance judgments.

---

## Project Overview

The goal of this project is to retrieve the most relevant prompts for a natural language user query.

A simple keyword search can work when the query and prompt use the same words, but it often fails when the wording differs. This project therefore combines multiple retrieval techniques:

- **Keyword retrieval** using TF-IDF
- **Semantic retrieval** using sentence-transformer embeddings and ChromaDB
- **Hybrid retrieval** combining keyword and semantic scores
- **Cross-encoder reranking** to improve candidate ordering
- **Metadata-aware scoring** using popularity and quality signals such as upvotes, likes, uses, views, fork count, author reputation, and downvotes

The final system is designed so different configurations can be tested and compared through the same pipeline.

---

## Repository Structure

```text
LEAF_Semantic_Prompt_Search/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── dataset.json
│   ├── evaluation_queries.json
│   ├── relevance_judgments.csv
│   ├── completed_relevance_judgments.csv
│   ├── retrieval_runs.csv
│   └── evaluation_results.csv
│
├── figures/
│   ├── top_experiments_ndcg_at_10.png
│   ├── pipeline_family_ndcg_at_10.png
│   ├── embedding_model_comparison_ndcg_at_10.png
│   ├── reranker_comparison_ndcg_at_10.png
│   ├── latency_by_experiment.png
│   └── ndcg_vs_latency.png
│
├── chroma/
│   └── ChromaDB persistent vector indexes
│
└── └── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    │
    ├── embedders.py
    ├── vectorstores.py
    ├── keyword_retriever.py
    ├── retrievers.py
    │
    ├── rerankers.py
    ├── metadata_scoring.py
    ├── pipelines.py
    │
    ├── build_indexes.py
    ├── generate_labeling_file.py
    ├── evaluation.py
    ├── plot_results.py
    └── demo_streamlit.py
```

---

## Python Files

### Configuration

`config.py`

Central configuration file. It defines paths, model names, retrieval settings, metadata weights, the `SearchConfig` dataclass, and the `get_config()` helper function.

### Data Loading and Preprocessing

`data_loader.py`  
`preprocessing.py`

`data_loader.py` loads the raw JSON dataset.

`preprocessing.py` converts raw prompt records into processed records with:

```text
id
text
metadata
```

### Retrieval Components

`embedders.py`  
`vectorstores.py`  
`keyword_retriever.py`  
`retrievers.py`

`embedders.py` wraps sentence-transformer embedding models.

`vectorstores.py` handles ChromaDB storage and semantic vector search.

`keyword_retriever.py` implements TF-IDF keyword retrieval.

`retrievers.py` coordinates semantic, keyword, and hybrid retrieval.

### Reranking, Metadata Scoring, and Pipeline

`rerankers.py`  
`metadata_scoring.py`  
`pipelines.py`

`rerankers.py` implements cross-encoder reranking.

`metadata_scoring.py` applies metadata-aware scoring.

`pipelines.py` combines retrieval, optional reranking, and optional metadata scoring into one end-to-end search pipeline.

### Executable Scripts

`build_indexes.py`  
`generate_labeling_file.py`  
`evaluation.py`  
`plot_results.py`

`build_indexes.py` builds Chroma vector indexes.

`generate_labeling_file.py` creates relevance judgment and retrieval run files.

`evaluation.py` computes evaluation metrics.

`plot_results.py` creates plots from the evaluation results.

`demo_streamlit` provides an interactive Streamlit interface for entering queries and viewing ranked prompt results.

---

## Main Components

### 1. Configuration

All major experiment settings are stored in:

```text
src/config.py
```

This file defines:

- dataset paths
- vector database path
- text template used for embeddings
- embedding model choices
- retrieval settings
- reranker model choices
- metadata scoring weights
- the `SearchConfig` dataclass
- the `get_config()` helper function

The project currently supports two embedding models:

```python
"minilm": "sentence-transformers/all-MiniLM-L6-v2"
"bge_small": "BAAI/bge-small-en-v1.5"
```

It also supports two cross-encoder rerankers:

```python
"msmarco": "cross-encoder/ms-marco-MiniLM-L-6-v2"
"tinybert": "cross-encoder/ms-marco-TinyBERT-L-2-v2"
```

---

### 2. Data Loading

Raw prompt data is loaded from:

`data/dataset.json`

using:

`src/data_loader.py`

The dataset is expected to be a list of dictionaries, where each dictionary represents one prompt.

---

### 3. Preprocessing

Preprocessing is handled in:

`src/preprocessing.py`

Each raw prompt record is transformed into a processed record with:

```python
{
    "id": "...",
    "text": "...",
    "metadata": {...}
}
```

The `text` field is built from the prompt's title, category, subcategory, tags, difficulty, and content.

Example embedded text:

```text
Title: Debug Python with PDB
Category: debugging
Subcategory: python-debugging
Tags: python, debugging, pdb, recursion
Difficulty: intermediate

Prompt:
...
```

This design gives the embedding model both the prompt content and useful contextual fields.

Metadata is stored separately for display, filtering, evaluation, and metadata-aware scoring.

---

### 4. Embeddings

Embeddings are generated in:

`src/embedders.py`

The `SentenceTransformerEmbedder` class supports:

- batch document embedding
- query embedding
- normalized embeddings for cosine-based retrieval

Batching is used to avoid loading the entire dataset into the model at once.

---

### 5. Vector Store

Semantic embeddings are stored in ChromaDB through:

`src/vectorstores.py`

The project uses a persistent local Chroma database located at:

`chroma/`

Each embedding model gets its own Chroma collection:

```text
prompts_minilm
prompts_bge_small
```

This prevents embeddings from different models from being mixed.

---

### 6. Keyword Retrieval

Keyword retrieval is implemented in:

`src/keyword_retriever.py`

It uses:

- `TfidfVectorizer`
- cosine similarity
- unigrams and bigrams
- English stop-word removal

This provides a lexical baseline and also contributes to hybrid retrieval.

Keyword retrieval compares the query and prompt texts using sparse TF-IDF vectors. This is different from semantic retrieval, which compares dense sentence-transformer embeddings.

---

### 7. Semantic, Keyword, and Hybrid Retrieval

Retrieval orchestration is handled in:

`src/retrievers.py`

The system supports three retrieval modes:

```python
"semantic"
"keyword"
"hybrid"
```

These are retrieval strategies, not different distance metrics.

- **Semantic retrieval** embeds the query and searches the Chroma vector database using cosine distance over dense embeddings.
- **Keyword retrieval** uses TF-IDF and cosine similarity over sparse lexical vectors.
- **Hybrid retrieval** combines semantic and keyword retrieval results.

In hybrid mode, the system retrieves candidates from both semantic search and keyword search, normalizes their scores, merges results by prompt ID, and computes:

```python
hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score
```

The default hybrid weights are:

```python
semantic_weight = 0.70
keyword_weight = 0.30
```

Semantic retrieval receives more weight because it better captures user intent, while keyword retrieval still helps with exact terms such as programming languages, tools, and error names.

---

### 8. Cross-Encoder Reranking

Reranking is implemented in:

`src/rerankers.py`

The reranker receives a query and candidate prompt text, then scores each query-prompt pair using a cross-encoder model.

The reranker is applied after first-stage retrieval. This design is intentional: the retrieval stage produces a larger candidate set, and the reranker selects the best final results from that candidate pool.

For evaluation, the system uses:

```python
semantic_top_k = 50
keyword_top_k = 50
merged_top_k = 75
final_top_k = 10
```

This means the first-stage retrievers collect many possible candidates, while the reranker has space to reorder and filter the final top 10 results.

---

### 9. Metadata-Aware Scoring

Metadata scoring is implemented in:

`src/metadata_scoring.py`

The metadata score uses engagement and quality signals:

```python
METADATA_SCORE_FIELDS = {
    "upvotes": 0.30,
    "likes": 0.22,
    "uses": 0.22,
    "fork_count": 0.10,
    "author_reputation": 0.10,
    "views": 0.06,
    "downvotes": -0.10,
}
```

The positive weights sum to 1.0. Downvotes are treated as a penalty.

Large count-based values are log-scaled with `log1p()` so that very large values such as views do not dominate the score.

The final score after metadata scoring is:

```python
final_score = search_score_weight * search_score + metadata_score_weight * metadata_score
```

The default weights are:

```python
search_score_weight = 0.85
metadata_score_weight = 0.15
```

This keeps relevance as the main ranking signal while allowing metadata to provide a small quality adjustment.

---

### 10. End-to-End Pipeline

The full search pipeline is implemented in:

`src/pipelines.py`

The pipeline builds the required components based on the selected configuration:

- keyword retriever if retrieval mode is `keyword` or `hybrid`
- embedder and Chroma vector store if retrieval mode is `semantic` or `hybrid`
- reranker if enabled
- metadata scorer if enabled

The general pipeline is:

```text
query
→ retrieval
→ optional cross-encoder reranking
→ optional metadata-aware scoring
→ final ranked results
```

---

## Installation

This project was developed with:

```text
Python 3.12.4
```

Create and activate a virtual environment if desired.

Then install dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
chromadb==1.5.9
sentence-transformers==5.4.1
scikit-learn==1.7.2
pandas==2.2.2
matplotlib==3.8.4
streamlit==1.32.0
```

---

## Building the Vector Indexes

Before semantic or hybrid search can run, the Chroma vector indexes must be built.

Run:

```bash
python src/build_indexes.py
```

This builds indexes for both embedding models:

```text
prompts_minilm
prompts_bge_small
```

Each collection stores:

- prompt IDs
- processed text
- metadata
- embedding vectors

The Chroma database is stored locally in:

`chroma/`

This folder should generally not be committed to Git.

---

## Running the Streamlit Demo

## Running the Streamlit Demo

The project includes an interactive Streamlit demo for running the search system.

Before running the demo, build the vector indexes:

```bash
python src/build_indexes.py
```

Then start the Streamlit app:

```bash
streamlit run src/demo_streamlit.py
```

The demo allows the user to:

- enter a natural language query
- choose the embedding model
- choose the retrieval mode
- enable or disable reranking
- choose the reranker model
- enable or disable metadata-aware scoring
- select the number of final results

The app returns ranked prompt results with their metadata and available scores.

---

## Evaluation Workflow

The project uses manual relevance judgments rather than automatic tag/category matching.

This was chosen because metadata fields such as tags and categories are not reliable enough to determine relevance. Two prompts may share a tag but satisfy different intents, and a relevant prompt may not always share the same tag as the query.

The evaluation process has four steps.

---

### Step 1: Create Evaluation Queries

Evaluation queries are stored in:

`data/evaluation_queries.json`

The queries were designed to be realistic user searches while remaining within the scope of the provided prompt dataset. Some queries are intentionally broad so that the evaluation can distinguish between highly relevant, partially relevant, and irrelevant results.

Example queries include:

```text
help me make my code better
write something persuasive for customers
help me understand the important parts of a contract
teach me the basics of machine learning
translate text from English to Italian
```

---

### Step 2: Generate Labeling and Retrieval Files

Run:

```bash
python src/generate_labeling_file.py
```

This creates two files:

```text
data/relevance_judgments.csv
data/retrieval_runs.csv
```

`relevance_judgments.csv` contains one row per unique query-result pair. It is used for manual labeling.

Labels are:

```text
0 = not relevant
1 = somewhat relevant
2 = highly relevant
```

`retrieval_runs.csv` contains one row per query, experiment, result, rank, score values, and latency measurement. It is used later to compute evaluation metrics and average query latency.

Latency is measured as the wall-clock time required for one full call to the search pipeline for a given query and experiment. This includes retrieval and, when enabled, reranking and metadata-aware scoring.

The reason for separating these files is that relevance depends only on:

`query + result_id`

not on which experiment retrieved the result. If multiple systems retrieve the same result for the same query, it only needs to be labeled once.

---

### Step 3: Complete Manual Labels

After labeling, save the completed file as:

`data/completed_relevance_judgments.csv`

The evaluation script uses this completed file.

---

### Step 4: Evaluate Experiments

Run:

```bash
python src/evaluation.py
```

This reads:

```text
data/completed_relevance_judgments.csv
data/retrieval_runs.csv
```

and outputs:

`data/evaluation_results.csv`

The evaluation computes:

- **Precision@5**
- **Precision@10**
- **MRR**
- **nDCG@5**
- **nDCG@10**
- **Average latency per query**

The main ranking-quality metric is **nDCG@10**, because the relevance labels are graded. Unlike precision, nDCG rewards systems that rank highly relevant results above somewhat relevant results. Average latency is reported as an efficiency metric to compare the speed cost of different retrieval and reranking configurations.

---

## Experiments

The experiment set compares:

- keyword retrieval
- MiniLM semantic retrieval
- BGE-small semantic retrieval
- semantic retrieval with reranking
- semantic retrieval with reranking and metadata
- hybrid retrieval
- hybrid retrieval with reranking
- hybrid retrieval with reranking and metadata
- MS MARCO reranker
- TinyBERT reranker

The strongest systems combine:

```text
hybrid retrieval
+ cross-encoder reranking
+ metadata-aware scoring
```

The best-performing setup in the final evaluation was:

`BGE-small + hybrid retrieval + MS MARCO reranker + metadata scoring`

---

## Plotting Results

After running evaluation, generate plots with:

```bash
python src/plot_results.py
```

This creates figures in:

`figures/`

Generated plots include:

```text
top_experiments_ndcg_at_10.png
pipeline_family_ndcg_at_10.png
embedding_model_comparison_ndcg_at_10.png
reranker_comparison_ndcg_at_10.png
latency_by_experiment.png
ndcg_vs_latency.png
```

The most useful plots are the pipeline-family comparison and the nDCG-versus-latency plot. The pipeline-family plot shows how each design choice affects ranking quality, while the latency plot shows the tradeoff between retrieval quality and runtime.

---

## Main Findings

The evaluation showed several clear patterns:

1. **Keyword retrieval was the weakest baseline.**  
   It worked for exact lexical matches, but it was less effective for broader natural language queries.

2. **Semantic retrieval improved over keyword search.**  
   Dense embeddings helped retrieve prompts that matched the meaning of the query even when wording differed.

3. **Cross-encoder reranking provided a large quality improvement.**  
   Reranking helped reorder the larger candidate set and improved final result quality.

4. **Metadata-aware scoring provided a smaller but useful boost.**  
   Metadata helped distinguish between similarly relevant prompts using popularity and trust signals.

5. **MS MARCO MiniLM performed better than TinyBERT in the full pipeline.**  
   TinyBERT was faster and lighter, but the MS MARCO MiniLM reranker produced stronger ranking quality.

6. **MiniLM and BGE-small were close in final performance.**  
   BGE-small achieved the best full-pipeline score, but MiniLM remained competitive.
   
7. **Reranking improved quality but increased latency.**  
   The latency measurements show the runtime cost of cross-encoder reranking. This makes the final system evaluation a quality-efficiency tradeoff rather than only a relevance comparison.

---

## Design Choices

### Why use a combined text template?

The embedding model receives a single string, not a dictionary. Therefore, the system combines important textual fields into one template:

```text
Title
Category
Subcategory
Tags
Difficulty
Prompt content
```

This gives the embedding model both the original prompt and contextual metadata that may help retrieval.

---

### Why use separate metadata?

Fields such as likes, upvotes, views, uses, downvotes, and author reputation are not part of the semantic meaning of the prompt. They are stored as metadata and used later for metadata-aware ranking.

This separation keeps semantic retrieval focused on meaning while allowing the final ranking stage to incorporate quality and popularity signals.

---

### Why use `top_k = 50` but `final_top_k = 10`?

The first-stage retrievers collect a broader candidate pool. The reranker then has enough results to compare and reorder.

If retrieval only returned 10 candidates and the final output also showed 10, the reranker could only reorder the same limited set. By retrieving 50 candidates and returning 10, the reranker can remove weaker candidates and promote stronger ones.

---

### Why use hybrid weights of 0.70 semantic and 0.30 keyword?

Semantic search is the main retrieval method because it captures query intent beyond exact word overlap. Keyword search still helps when exact terms are important, such as programming languages, tools, or error names.

The 0.70 / 0.30 split gives semantic retrieval the main influence while preserving keyword evidence.

---

### Why use metadata weights?

The metadata fields were weighted by how directly they reflect prompt quality:

```text
upvotes: strongest community approval signal
likes and uses: strong quality/usefulness signals
fork_count and author reputation: moderate reuse/trust signals
views: weaker popularity signal
downvotes: negative quality signal
```

The final metadata score is only given a 15% influence so that popularity does not override relevance.

---

## Reproducing the Full Workflow

From a clean project state:

```bash
pip install -r requirements.txt
```

Build indexes:

```bash
python src/build_indexes.py
```
Run the Streamlit demo after building indexes if you only want to test the search system interactively. The evaluation workflow is only needed to reproduce the reported metrics.

Run the interactive demo:

```bash
streamlit run src/demo_streamlit.py
```

Generate retrieval runs and labeling file:

```bash
python src/generate_labeling_file.py
```

Manually fill:

`data/relevance_judgments.csv`

Save completed labels as:

`data/completed_relevance_judgments.csv`

Run evaluation:

```bash
python src/evaluation.py
```

Generate plots:

```bash
python src/plot_results.py
```

---

## Notes

- The `chroma/` folder contains generated vector indexes and should generally be ignored by Git.
- The raw dataset may be too large to commit depending on project submission rules.
- If the dataset or preprocessing template changes, the Chroma indexes should be rebuilt.
- If the embedding model changes, a separate Chroma collection is required.
- If experiment names change but query-result pairs remain the same, manual relevance labels do not need to be redone.
- The Streamlit demo requires `streamlit` to be installed and the Chroma indexes to exist.
- The reported latency values are measured during evaluation and may vary depending on hardware.
- Manual relevance labeling is not required to reproduce the existing results because `completed_relevance_judgments.csv` is already included.

---

## Future Work

Possible extensions include:

- fine-tuning an embedding model on prompt-specific relevance data
- learning metadata weights from user feedback
- adding query filters for category, language, difficulty, or target model
- saving the TF-IDF keyword index to disk
- measuring latency separately for retrieval, reranking, and metadata scoring
- improving the Streamlit interface with filters, saved searches, and result export
- collecting real user click feedback for learning-to-rank
