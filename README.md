# Improving Retrieval for Financial Question Answering

Code and evaluation framework for the SIGIR 2026 paper submission:
**"Improving Retrieval in Financial Question Answering with Query Expansion and Reranking"**

We benchmark three zero-shot retrieval strategies on the
[FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) dataset
of financial 10-K documents and evaluate answers with cosine similarity, BERTScore,
and LLM-based correctness scoring.

---

## Methods

| Method | Description |
|---|---|
| **Base RAG** | Dense retrieval (top-5 chunks) with `text-embedding-ada-002`, answered by GPT-4o |
| **Query Expansion** | Two-stage: generate initial answer without context, then retrieve using query + initial answer |
| **Reranker** | Retrieve top-10 chunks, rerank with CrossEncoder (`ms-marco-TinyBERT-L-2-v2`), answer with top-3 |

## Evaluation Metrics

- **Cosine Similarity** — TF-IDF cosine similarity between model answer and reference answer
- **BERTScore** — Precision score using contextual BERT embeddings
- **LLM Eval** — GPT-4o assigns a 0–1 correctness score with justification

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd RetreivalRepo
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and set your OpenAI API key
```

### 3. Run experiments

```bash
# Individual experiments
python -m src.base_rag
python -m src.query_expansion
python -m src.reranker

# All experiments sequentially
bash run_experiments.sh
```

Results are saved to `results/`.

---

## Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── utils.py          # PDF handling, Chroma DB creation, OpenAI call wrapper
│   ├── evaluation.py     # Cosine similarity, BERTScore, LLM-based evaluation
│   ├── base_rag.py       # Baseline RAG experiment (~60 lines)
│   ├── query_expansion.py # Query expansion experiment (~70 lines)
│   └── reranker.py       # CrossEncoder reranker experiment (~70 lines)
├── configs/
│   └── config.yaml       # Model names, top-k values, prompt templates
├── results/              # CSV output files (generated at runtime)
├── .env.example          # Template for API key configuration
├── requirements.txt
├── run_experiments.sh
└── README.md
```

---

## Results

Results CSVs are written to `results/` with columns:

| Column | Description |
|---|---|
| `doc_name` | Source document name |
| `question` | FinanceBench question |
| `ref_answer` | Ground-truth reference answer |
| `model_answer` | Model-generated answer |
| `cosine_similarity` | TF-IDF cosine similarity (0–1) |
| `bert_score` | BERTScore precision (0–1) |
| `llm_eval` | GPT-4o correctness score (0–1) |

---

## Notes

- `pdf_documents/` and `vector_db/` are excluded from git (large binary files regenerated at runtime).
- The in-memory Chroma client is re-created per document to avoid collection naming collisions.
- Set `num_samples` in `configs/config.yaml` to run on more FinanceBench examples.
