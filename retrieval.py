import os
import json
import time
from typing import List, Dict, NamedTuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi

# Directory where JSON chunks are stored
CHUNKS_DIR = "sec_chunks"

# Lazy-initialized optional components
_qdrant = None
_model = None
_bm25 = None
_bm25_ids = []
_bm25_metadata = []


def _init_bm25():
    global _bm25, _bm25_ids, _bm25_metadata
    if _bm25 is not None:
        return
    corpus = []
    ids = []
    metadata = []
    if not os.path.isdir(CHUNKS_DIR):
        raise RuntimeError(f"Chunks directory not found: {CHUNKS_DIR}")
    for fname in os.listdir(CHUNKS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CHUNKS_DIR, fname)
        data = json.load(open(path, "r", encoding="utf-8"))
        chunk_id = data.get("source_chunk_id", fname.replace('.json', ''))
        ids.append(chunk_id)
        metadata.append(data)
        corpus.append(data.get("text", "").split())
    _bm25_ids = ids
    _bm25_metadata = metadata
    _bm25 = BM25Okapi(corpus)


def _init_vector_components():
    """Try to initialize Qdrant client and embedding model. Fail quietly so BM25-only mode still works."""
    global _qdrant, _model
    if _qdrant is not None and _model is not None:
        return
    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
    except Exception:
        # Missing optional dependencies; vector search won't be available
        return
    try:
        _qdrant = QdrantClient(url="http://localhost:6333")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        # If Qdrant not running or model download fails, keep BM25 available
        _qdrant = None
        _model = None

# Doc type to hold retrieval results
class Doc(NamedTuple):
    id: str
    text: str
    score: float
    metadata: Dict

# Vector search implementation
def vector_search(query: str, k: int) -> List[Doc]:
    """
    Perform a vector search in Qdrant:
      1. Encode the query text to a vector.
      2. Query Qdrant for the top-k most similar vectors.
      3. Load text and metadata for each result.
    """
    # Ensure vector components are available
    _init_vector_components()
    if _qdrant is None or _model is None:
        # Vector search not available; return empty list so BM25 can be used alone
        return []
    query_vec = _model.encode(query).tolist()
    results = _qdrant.search(
        collection_name="finrag_chunks",
        query_vector=query_vec,
        limit=k
    )
    docs = []
    for point in results:
        payload = getattr(point, "payload", {}) or {}
        source_id = payload.get("source_chunk_id", str(getattr(point, "id", "")))
        file_path = os.path.join(CHUNKS_DIR, f"{source_id}.json")
        try:
            data = json.load(open(file_path, "r", encoding="utf-8"))
            text = data.get("text", "")
        except Exception:
            text = ""
        score = getattr(point, "score", 0.0)
        docs.append(Doc(
            id=source_id,
            text=text,
            score=score,
            metadata=payload
        ))
    return docs

# BM25 search implementation
def bm25_search(query: str, k: int) -> List[Doc]:
    """
    Perform a BM25 keyword search over the text chunks.
    """
    _init_bm25()
    tokenized = query.split()
    scores = _bm25.get_scores(tokenized)
    top_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    docs = []
    for idx, score in top_indices:
        metadata = _bm25_metadata[idx]
        # Prefer the original stored text in metadata (more reliable)
        text = metadata.get("text", " ")
        docs.append(Doc(
            id=_bm25_ids[idx],
            text=text,
            score=score,
            metadata=metadata
        ))
    return docs

# Remaining fusion logic unchanged...

def _rank_map(ids: List[str]) -> Dict[str, int]:
    return {doc_id: rank for rank, doc_id in enumerate(ids, start=1)}

def _rrf_fuse(
    vec_ids: List[str],
    bm25_ids: List[str],
    w_vec: float = 1.0,
    w_bm25: float = 1.0,
    K: int = 60
) -> Dict[str, float]:
    rv = _rank_map(vec_ids)
    rb = _rank_map(bm25_ids)
    all_ids = set(vec_ids) | set(bm25_ids)
    big_rank = max(len(vec_ids), len(bm25_ids)) + 1
    fused = {}
    for doc_id in all_ids:
        rank_v = rv.get(doc_id, big_rank)
        rank_b = rb.get(doc_id, big_rank)
        fused[doc_id] = w_vec / (K + rank_v) + w_bm25 / (K + rank_b)
    return fused

def retrieve(
    query: str,
    k: int = 10,
    k_each: int = 50,
    w_vec: float = 1.0,
    w_bm25: float = 1.0
) -> List[Doc]:
    # 1) Get top results
    vec_docs = vector_search(query, k_each)
    bm_docs = bm25_search(query, k_each)
    vec_ids = [d.id for d in vec_docs]
    bm_ids = [d.id for d in bm_docs]
    # 2) Fuse ranks
    fused = _rrf_fuse(vec_ids, bm_ids, w_vec=w_vec, w_bm25=w_bm25, K=k_each)
    # 3) Merge and rerank
    doc_map = {d.id: d for d in vec_docs}
    for d in bm_docs:
        doc_map.setdefault(d.id, d)
    sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    return [Doc(id=i, text=doc_map[i].text, score=fused[i], metadata=doc_map[i].metadata) for i,_ in sorted_ids]
