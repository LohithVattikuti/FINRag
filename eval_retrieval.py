import json
import os
import time
import numpy as np
import mlflow
from typing import List, Tuple
from retrieval import retrieve

# Debug mode flag (turn off for production runs)
DEBUG = True

# Path to your labeled queries file (relative to repository root)
EVAL_PATH = "eval_queries.json"


def load_eval(path: str) -> List[Tuple[str, List[str]]]:
    """
    Reads JSON like:
      [{ "query": "…", "positives": ["id1","id2"] }, …]
    Returns a list of (query, [positive_id,…]) tuples.
    This is robust to common errors and prints helpful debug info when DEBUG=True.
    """
    base_dir = os.path.dirname(__file__)
    # Accept absolute paths as well as repository-relative paths
    full_path = path if os.path.isabs(path) else os.path.join(base_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Evaluation file not found: {full_path}")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        # Provide a clearer error message for malformed JSON
        raise RuntimeError(f"Failed to load or parse evaluation JSON at {full_path}: {e}")

    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {full_path}, got {type(records)}")

    parsed = []
    for i, r in enumerate(records):
        if not isinstance(r, dict) or "query" not in r:
            raise ValueError(f"Record {i} is malformed; each record must be an object with a 'query' key")
        positives = r.get("positives", [])
        if positives is None:
            positives = []
        parsed.append((r["query"], positives))

    if DEBUG:
        print(f"DEBUG: Loaded {len(parsed)} evaluation records from: {full_path}")

    return parsed

def ndcg_at_k(rel: List[int], k: int = 10) -> float:
    """
    NDCG@k based on binary relevance list (1 if in positives, else 0).
    """
    def dcg(scores):
        return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))
    ideal = sorted(rel, reverse=True)[:k]
    return (dcg(rel[:k]) / dcg(ideal)) if any(ideal) else 0.0

def reciprocal_rank(pred_ids: List[str], gold_ids: List[str]) -> float:
    """
    MRR: 1 / rank of first correct hit, or 0 if none.
    """
    gold = set(gold_ids)
    for idx, pid in enumerate(pred_ids, start=1):
        if pid in gold:
            return 1.0 / idx
    return 0.0

def recall_at_k(pred_ids: List[str], gold_ids: List[str]) -> float:
    """
    Recall@k: fraction of gold_ids retrieved in the top-k.
    """
    if not gold_ids:
        return 0.0
    return len(set(pred_ids) & set(gold_ids)) / len(gold_ids)

def estimate_cost(num_queries: int) -> float:
    """
    Placeholder cost estimator. Adjust to your pricing.
    """
    cost_per_query = 0.0008  # e.g., $0.0008/query
    return cost_per_query * num_queries

def run_evaluation(k: int = 10):
    """
    Main loop:
    - Loads labeled set
    - Runs retrieve() on each query
    - Computes all metrics
    - Logs to MLflow under a single run
    """
    eval_set = load_eval(EVAL_PATH)
    latencies, ndcgs, mrrs, recs = [], [], [], []

    with mlflow.start_run(run_name="hybrid_retrieval_eval"):
        for query, positives in eval_set:
            start = time.time()
            docs = retrieve(query, k=k)
            elapsed = time.time() - start

            pred_ids = [d.id for d in docs]
            # build binary relevance list
            rel = [1 if pid in positives else 0 for pid in pred_ids]

            latencies.append(elapsed)
            ndcgs.append(ndcg_at_k(rel, k))
            mrrs.append(reciprocal_rank(pred_ids, positives))
            recs.append(recall_at_k(pred_ids, positives))

        # Log aggregate metrics
        mlflow.log_metric("NDCG_10_mean", float(np.mean(ndcgs)))
        mlflow.log_metric("MRR_mean", float(np.mean(mrrs)))
        mlflow.log_metric("Recall_k_mean", float(np.mean(recs)))
        mlflow.log_metric("latency_P50_ms", float(np.percentile(latencies, 50) * 1000))
        mlflow.log_metric("latency_P95_ms", float(np.percentile(latencies, 95) * 1000))
        mlflow.log_metric("cost_per_100_queries_usd", estimate_cost(len(eval_set)))

if __name__ == "__main__":
    run_evaluation()
