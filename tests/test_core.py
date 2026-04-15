import os
import sys
import json
import tempfile
# ensure repo root is on sys.path for imports when running under pytest
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from retrieval import bm25_search
from eval_retrieval import load_eval


def test_load_eval_reads_file(tmp_path):
    data = [
        {"query": "revenue growth", "positives": ["AAPL_10-K_2022-10-28_chunk0"]}
    ]
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    records = load_eval(str(p))
    assert isinstance(records, list)
    assert records[0][0] == "revenue growth"
    assert isinstance(records[0][1], list)


def test_bm25_search_returns_docs(tmp_path, monkeypatch):
    # prepare a temporary sec_chunks folder with one chunk
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    chunks_dir = repo_root / "sec_chunks"
    chunks_dir.mkdir()
    chunk = {
        "source_chunk_id": "test_chunk",
        "text": "company reported strong revenue growth in the quarter"
    }
    (chunks_dir / "test_chunk.json").write_text(json.dumps(chunk), encoding="utf-8")

    # monkeypatch the CHUNKS_DIR used by retrieval module
    import retrieval
    monkeypatch.setattr(retrieval, "CHUNKS_DIR", str(chunks_dir))

    docs = bm25_search("revenue growth", k=1)
    assert len(docs) >= 1
    assert "revenue" in docs[0].text
