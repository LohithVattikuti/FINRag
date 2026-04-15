# FinRag (Financial Retrieval & Augmented Generation)

This repository contains tools to:

- Download SEC filings (EDGAR) as HTML (`data_ingestion/sec_scraper.py`).
- Parse and chunk filings into JSON text chunks (`data_ingestion/parse_and_chunk.py`).
- Create embeddings and upsert into a local Qdrant vector DB (`embeddings/generate_embeddings.py`).
- Hybrid retrieval (BM25 + vector fusion) and an evaluation harness (`retrieval.py`, `eval_retrieval.py`).

Quick start
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. (Optional) Run the SEC downloader to fetch filings:

```bash
python data_ingestion/sec_scraper.py
```

4. Parse and chunk the HTML files:

```bash
python data_ingestion/parse_and_chunk.py
```

5. (Optional) Start a local Qdrant server (Docker recommended) and run embedding ingest:

```bash
# start qdrant with docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
python embeddings/generate_embeddings.py
```

6. Evaluate retrieval (works in BM25-only mode if Qdrant/model missing):

```bash
python eval_retrieval.py
```

Notes
- The system is designed to be usable without Qdrant or heavy models — BM25 will work on the JSON chunks alone.
- `eval_queries.json` contains example queries and labeled positives used by `eval_retrieval.py`.

Contributions
Feel free to open issues or PRs to improve ingestion, evaluation, or add Docker-based dev setup.
