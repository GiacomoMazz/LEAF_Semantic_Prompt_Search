# Source Code Run Order

Run all commands from the project root unless stated otherwise.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python version used: `3.12.4`

## 2. Build vector indexes

```bash
python src/build_indexes.py
```

This builds the Chroma collections for both embedding models:

```text
prompts_minilm
prompts_bge_small
```

## 3. Run the Streamlit demo

```bash
streamlit run src/demo_streamlit.py
```

The demo provides an interactive interface for entering a natural language query and returning ranked prompt results.

## 4. Generate evaluation files

```bash
python src/generate_labeling_file.py
```

This creates:

```text
data/relevance_judgments.csv
data/retrieval_runs.csv
```

## 5. Complete manual relevance labels

This step is only needed if regenerating the evaluation files or changing the queries, dataset, or experiment configurations.

The repository already includes:

```text
data/completed_relevance_judgments.csv
data/retrieval_runs.csv
data/evaluation_results.csv
```

If new labels are needed, fill the `relevance` column in:

`data/relevance_judgments.csv`

Then save the completed file as:

`data/completed_relevance_judgments.csv`

Labels:

```text
0 = not relevant
1 = somewhat relevant
2 = highly relevant
```

## 6. Run evaluation

```bash
python src/evaluation.py
```

This creates:

`data/evaluation_results.csv`

## 7. Generate plots

```bash
python src/plot_results.py
```

This creates the result figures in:

`figures/`

## Notes

- Semantic and hybrid search require the Chroma indexes to be built first.
- The Streamlit demo requires the vector indexes to already exist.
- If `dataset.json`, the preprocessing template, or embedding models change, rerun `build_indexes.py`.
- If only experiment names change but query-result pairs stay the same, manual relevance labels do not need to be redone.