"""
ContextCache benchmark on similiar_qqp dataset.

Mirrors the GPTCache benchmark structure for fair comparison:
  - Insert 300 origin questions into cache
  - Query with 1000 similar questions (300 should hit, 700 should not)
  - Report precision, recall, F1

Usage:
    python benchmark_contextcache.py                        # default settings
    python benchmark_contextcache.py --threshold 0.80
    python benchmark_contextcache.py --reset
"""

import argparse
import json
import os
import random
import shutil
import tarfile
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter.api import get as cache_get, put as cache_put


def context_pre_embedding(data, **_):
    """Return (query, None) 2-tuple as required by ContextCache adapter."""
    prompt = data.get("prompt")
    return prompt, None

CACHE_DIR   = "cache_contextcache_benchmark"
QQP_PATH    = "../GPTCache/examples/benchmark/similiar_qqp.json.gz"

EMBED_DIM   = 384
INSERT_SIZE = 300
TEST_SIZE   = 1000


def load_qqp(path):
    with open(path, "rb") as f:
        t = tarfile.open(fileobj=f)
        data = json.loads(t.extractfile(t.getmembers()[0]).read())
    return data


class MiniLMEmbedding:
    """Wrapper around SentenceTransformer for gptcache."""
    def __init__(self):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = EMBED_DIM

    def to_embeddings(self, text, **_):
        if isinstance(text, list):
            text = text[-1]
        emb = self._model.encode([text], normalize_embeddings=True,
                                  convert_to_numpy=True)[0]
        return emb.astype("float32")


class WrapEvaluation(SearchDistanceEvaluation):
    def evaluation(self, src_dict, cache_dict, **kwargs):
        return super().evaluation(src_dict, cache_dict, **kwargs)
    def range(self):
        return super().range()


def evaluate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def run(threshold=0.75, reset=False):
    random.seed(42)

    if reset and os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared {CACHE_DIR}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    sqlite_file = os.path.join(CACHE_DIR, "sqlite.db")
    faiss_file  = os.path.join(CACHE_DIR, "faiss.index")
    has_data    = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

    print("Loading QQP data...")
    all_pairs = load_qqp(QQP_PATH)
    random.shuffle(all_pairs)

    # Assign IDs
    for i, pair in enumerate(all_pairs):
        pair["id"] = str(i)

    insert_pairs = all_pairs[:INSERT_SIZE]
    test_pairs   = all_pairs[:TEST_SIZE]   # first INSERT_SIZE overlap → should hit

    print(f"  Insert pool: {INSERT_SIZE} pairs")
    print(f"  Test pool:   {TEST_SIZE} pairs ({INSERT_SIZE} positive, "
          f"{TEST_SIZE - INSERT_SIZE} negative)")

    # ── init cache ──
    emb = MiniLMEmbedding()
    cache_base   = CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_file}")
    vector_base  = VectorBase("faiss", dimension=EMBED_DIM, index_path=faiss_file)
    data_manager = get_data_manager(cache_base, vector_base, max_size=100000)

    cache.init(
        pre_embedding_func=context_pre_embedding,
        embedding_func=emb.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=WrapEvaluation(),
        config=Config(similarity_threshold=threshold),
    )

    # ── insert ──
    if not has_data:
        print(f"\nInserting {INSERT_SIZE} questions...")
        t0 = time.time()
        questions = [p["origin"] for p in insert_pairs]
        answers   = [p["id"]     for p in insert_pairs]
        cache.import_data(questions=questions, answers=answers)
        cache.data_manager.flush()
        print(f"Insert done in {time.time()-t0:.1f}s")

    # ── query ──
    print(f"\nRunning {TEST_SIZE} queries (threshold={threshold})...")
    random.shuffle(test_pairs)

    tp = fp = fn_pos = fn_neg = miss = 0
    all_time = 0.0
    t_query = time.time()

    for pair in test_pairs:
        is_positive = int(pair["id"]) < INSERT_SIZE   # was inserted into cache

        t0  = time.time()
        raw = cache_get(pair["similar"])
        elapsed = time.time() - t0
        all_time += elapsed

        # ContextCache adapt() returns (result, is_hit, ...) tuple
        if isinstance(raw, tuple):
            res, is_hit = raw[0], raw[1]
        else:
            res, is_hit = raw, raw is not None

        if is_hit and res is not None:
            # cache hit
            if res == pair["id"]:
                tp += 1                        # correct hit
            else:
                fp += 1                        # wrong hit
        else:
            # cache miss
            if is_positive:
                fn_pos += 1                    # should have hit but didn't
            else:
                fn_neg += 1                    # correctly missed (true negative)

    total_query_time = time.time() - t_query
    precision, recall, f1 = evaluate_metrics(tp, fp, fn_pos)

    print(f"\n{'='*55}")
    print(f"ContextCache Benchmark Results  (threshold={threshold})")
    print(f"{'='*55}")
    print(f"Total queries        : {TEST_SIZE}")
    print(f"  Positives (in cache): {INSERT_SIZE}")
    print(f"  Negatives (not in)  : {TEST_SIZE - INSERT_SIZE}")
    print(f"Cache hit correct (TP): {tp}")
    print(f"Cache hit wrong   (FP): {fp}")
    print(f"Missed positives  (FN): {fn_pos}")
    print(f"Correct miss      (TN): {fn_neg}")
    print(f"{'─'*55}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1 Score             : {f1:.4f}")
    print(f"{'─'*55}")
    print(f"Avg query latency    : {all_time/TEST_SIZE*1000:.1f}ms")
    print(f"Total query time     : {total_query_time:.1f}s")
    print(f"Avg embed time       : {cache.report.average_embedding_time()}")
    print(f"Avg search time      : {cache.report.average_search_time()}")

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn_pos, "tn": fn_neg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run(threshold=args.threshold, reset=args.reset)
