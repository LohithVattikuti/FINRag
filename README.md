
# FINRag — Financial Retrieval & Augmented Generation

FINRag (Financial Retrieval & Augmented Generation) is a compact toolkit and reference pipeline for building information retrieval over SEC filings and other financial documents. It combines simple, robust keyword retrieval (BM25) with optional vector search (embeddings stored in Qdrant) and provides an evaluation harness to measure retrieval quality.

Repository: https://github.com/LohithVattikuti/FINRag

![CI](https://github.com/LohithVattikuti/FINRag/actions/workflows/ci.yml/badge.svg)

Key components
- data_ingestion/sec_scraper.py — download filings from SEC EDGAR (HTML)
- data_ingestion/parse_and_chunk.py — convert HTML -> plain text and split into overlapping JSON chunks
- embeddings/generate_embeddings.py — generate sentence-transformer embeddings and upsert to Qdrant
- retrieval.py — BM25 + optional vector retrieval and rank fusion logic
- eval_retrieval.py — evaluation harness (NDCG, MRR, Recall) and MLflow logging

Why this project
- Lightweight: works out-of-the-box in BM25-only mode (no vector DB required).
- Extensible: optional vector search path using Qdrant + sentence-transformers for semantic retrieval.
- Reproducible: CI and tests included to help maintain code quality.

Quick start (30–60 seconds)
1. Create and activate a Python 3.10+ virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. If you already have SEC HTML files in `sec_filings/`, create chunks:

```bash
python data_ingestion/parse_and_chunk.py
```

4. Run BM25 retrieval directly (no external services required):

```python
from retrieval import retrieve
docs = retrieve("revenue growth", k=5)
for d in docs:
		print(d.id, d.score)
```

Optional: semantic vectors with Qdrant
1. Start Qdrant (Docker):

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

2. Generate and upsert embeddings:

```bash
python embeddings/generate_embeddings.py
```

3. Now `retrieve()` will use vector search (if available) and fuse BM25 + vector ranks.

Evaluation
- `eval_queries.json` contains labeled queries (list of {"query":..., "positives": [...]}) used by `eval_retrieval.py`.
- Run evaluation and log metrics via MLflow:

```bash
python eval_retrieval.py
```

Project architecture (high-level)
- Ingestion: download raw HTML filings from EDGAR, store under `sec_filings/`.
- Parsing & chunking: convert HTML to plain text and split into overlapping chunks (JSON files in `sec_chunks/`).
- Embeddings (optional): embed chunk text and upsert to Qdrant collection `finrag_chunks`.
- Retrieval: BM25 index on chunk text + optional vector search; fused ranking using Reciprocal Rank Fusion style scoring.
- Evaluation: scripted metrics (NDCG@k, MRR, Recall@k) with MLflow logging.

Testing and CI
- Minimal pytest-based unit tests are included under `tests/` and the repository includes a GitHub Actions workflow at `.github/workflows/ci.yml` to run tests on push/PR.

Contributing
- Contributions are welcome — please open an issue or submit a pull request.
- Suggested improvements:
	- Add Docker Compose for Qdrant + MLflow to streamline local development
	- Add end-to-end integration tests that run with a local Qdrant instance
	- Add richer evaluation scripts and dataset management utilities

License
- This project is provided under the MIT license. See `LICENSE`.

Contact
- Maintainer: Lohith Vattikuti (https://github.com/LohithVattikuti)

Thank you for using FINRag — if you'd like, I can add Docker Compose, extended CI (linting + types), or a simple CLI wrapper next.
