"""
Ablation study: GPTCache embedding variants vs ContextCache on similiar_qqp.

Design
------
Dataset : similiar_qqp.json.gz  (300 inserted, 1000 tested: 300 pos + 700 neg)
Metrics : Precision, Recall, F1, TP, FP, FN, TN, Avg-query-latency (ms)

GPTCache variants (SearchDistanceEvaluation):
  - ONNX     paraphrase-albert-onnx        768-dim
  - MiniLM   all-MiniLM-L6-v2              384-dim
  - BGE-S    BAAI/bge-small-en-v1.5        384-dim
  - BGE-B    BAAI/bge-base-en-v1.5         768-dim

ContextCache variant:
  - CC-MiniLM  MiniLM + trained BatchOptimizedEncoder

Thresholds: 0.70, 0.75, 0.80, 0.85, 0.90

Usage:
    python ablation_study.py
    python ablation_study.py --reset      # delete all caches and rebuild
    python ablation_study.py --out results/ablation.csv
"""

import argparse
import csv
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
from gptcache.adapter.api import get as cache_get

# ── paths ──────────────────────────────────────────────────────────────────────
QQP_PATH   = "../GPTCache/examples/benchmark/similiar_qqp.json.gz"
CACHE_ROOT = "cache_ablation"

INSERT_SIZE = 300
TEST_SIZE   = 1000
THRESHOLDS  = [0.70, 0.75, 0.80, 0.85, 0.90]

os.makedirs(CACHE_ROOT, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def load_qqp(path):
    with open(path, "rb") as f:
        t = tarfile.open(fileobj=f)
        data = json.loads(t.extractfile(t.getmembers()[0]).read())
    return data


def context_pre_embedding(data, **_):
    """ContextCache adapter expects a 2-tuple (query, context)."""
    return data.get("prompt"), None


def evaluate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def query_result(raw):
    """Parse ContextCache adapter's 4-tuple or plain return value."""
    if isinstance(raw, tuple):
        return raw[0], raw[1]       # (value, is_hit)
    return raw, raw is not None


# ── embedding wrappers ─────────────────────────────────────────────────────────

class SBERTEmbed:
    def __init__(self, model_name, dim):
        self._model = SentenceTransformer(model_name)
        self.dimension = dim
        self.name = model_name

    def to_embeddings(self, text, **_):
        if isinstance(text, list):
            text = text[-1]
        return self._model.encode([text], normalize_embeddings=True,
                                  convert_to_numpy=True)[0].astype("float32")


class OnnxEmbed:
    def __init__(self):
        from gptcache.embedding import Onnx
        self._model = Onnx()
        self.dimension = self._model.dimension
        self.name = "onnx-paraphrase-albert"

    def to_embeddings(self, text, **_):
        if isinstance(text, list):
            text = text[-1]
        return self._model.to_embeddings(text)


# ── per-embedding cache manager ───────────────────────────────────────────────

def build_cache(embed_obj, cache_dir, threshold):
    """Init (or re-use) a SQLite+FAISS cache for the given embedding."""
    os.makedirs(cache_dir, exist_ok=True)
    sqlite_file = os.path.join(cache_dir, "sqlite.db")
    faiss_file  = os.path.join(cache_dir, "faiss.index")
    has_data    = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

    cache_base   = CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_file}")
    vector_base  = VectorBase("faiss", dimension=embed_obj.dimension,
                              index_path=faiss_file)
    data_manager = get_data_manager(cache_base, vector_base, max_size=100_000)

    cache.init(
        pre_embedding_func=context_pre_embedding,
        embedding_func=embed_obj.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(similarity_threshold=threshold),
    )
    return has_data


def insert_questions(insert_pairs):
    questions = [p["origin"] for p in insert_pairs]
    answers   = [p["id"]     for p in insert_pairs]
    cache.import_data(questions=questions, answers=answers)
    cache.data_manager.flush()


def run_queries(test_pairs, threshold):
    cache.config.similarity_threshold = threshold
    tp = fp = fn_pos = fn_neg = 0
    total_time = 0.0
    t_query = time.time()

    for pair in test_pairs:
        is_positive = int(pair["id"]) < INSERT_SIZE
        t0  = time.time()
        raw = cache_get(pair["similar"])
        total_time += time.time() - t0

        res, is_hit = query_result(raw)

        if is_hit and res is not None:
            if res == pair["id"]:
                tp += 1
            else:
                fp += 1
        else:
            if is_positive:
                fn_pos += 1
            else:
                fn_neg += 1

    wall = time.time() - t_query
    avg_ms = total_time / len(test_pairs) * 1000
    return tp, fp, fn_pos, fn_neg, avg_ms, wall


# ── ContextCache variant ───────────────────────────────────────────────────────

def run_contextcache_variant(test_pairs, threshold, cache_dir):
    """
    ContextCache 2-stage evaluation:
      Stage 1 – query cosine similarity > similarity_threshold
      Stage 2 – context mean cosine similarity > dialuoge_threshold (0.6)

    NOTE: For single-turn QQP queries context_data is just the query embedding,
    so stage-2 equals stage-1.  The ContextCache adapter performs this inline and
    does NOT call similarity_evaluation, so we use SearchDistanceEvaluation here
    to avoid loading the heavy BatchOptimizedEncoder unnecessarily.
    The dialuoge_threshold=0.6 default is always satisfied when threshold >= 0.6.
    """
    os.makedirs(cache_dir, exist_ok=True)
    sqlite_file = os.path.join(cache_dir, "sqlite.db")
    faiss_file  = os.path.join(cache_dir, "faiss.index")
    has_data    = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

    embed_obj = SBERTEmbed("all-MiniLM-L6-v2", 384)

    cache_base   = CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_file}")
    vector_base  = VectorBase("faiss", dimension=384, index_path=faiss_file)
    data_manager = get_data_manager(cache_base, vector_base, max_size=100_000)

    cache.init(
        pre_embedding_func=context_pre_embedding,
        embedding_func=embed_obj.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),   # adapter bypasses this
        config=Config(similarity_threshold=threshold),      # 2-stage inline in adapter
    )

    if not has_data:
        insert_pairs = test_pairs[:INSERT_SIZE]
        print(f"  [CC-MiniLM] inserting {INSERT_SIZE} questions...")
        t0 = time.time()
        insert_questions(insert_pairs)
        print(f"  [CC-MiniLM] insert done in {time.time()-t0:.1f}s")

    return run_queries(test_pairs, threshold)


# ── main ───────────────────────────────────────────────────────────────────────

def main(reset=False, out_csv="results/ablation.csv"):
    random.seed(42)
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)

    if reset:
        if os.path.isdir(CACHE_ROOT):
            shutil.rmtree(CACHE_ROOT)
            print(f"Cleared {CACHE_ROOT}")
        os.makedirs(CACHE_ROOT)

    # ── load data ──
    print("Loading QQP data...")
    all_pairs = load_qqp(QQP_PATH)
    random.shuffle(all_pairs)
    for i, p in enumerate(all_pairs):
        p["id"] = str(i)

    insert_pairs = all_pairs[:INSERT_SIZE]
    test_pairs   = all_pairs[:TEST_SIZE]    # first INSERT_SIZE overlap → positives

    print(f"  Insert: {INSERT_SIZE}  |  Test: {TEST_SIZE} "
          f"({INSERT_SIZE} pos, {TEST_SIZE-INSERT_SIZE} neg)")

    # ── define GPTCache embedding variants ──
    print("\nLoading embedding models...")
    gptcache_variants = [
        ("GPTCache-ONNX",    OnnxEmbed()),
        ("GPTCache-MiniLM",  SBERTEmbed("all-MiniLM-L6-v2",       384)),
        ("GPTCache-BGE-S",   SBERTEmbed("BAAI/bge-small-en-v1.5", 384)),
        ("GPTCache-BGE-B",   SBERTEmbed("BAAI/bge-base-en-v1.5",  768)),
    ]

    rows = []

    # ── GPTCache variants × thresholds ──
    for variant_name, embed_obj in gptcache_variants:
        safe_name = variant_name.lower().replace(" ", "_").replace("-", "_")
        cache_dir = os.path.join(CACHE_ROOT, safe_name)

        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}  (dim={embed_obj.dimension})")
        print(f"{'='*60}")

        # Build cache once at a dummy threshold; insert if needed
        has_data = build_cache(embed_obj, cache_dir, threshold=THRESHOLDS[0])
        if not has_data:
            print(f"  Inserting {INSERT_SIZE} questions...")
            t0 = time.time()
            insert_questions(insert_pairs)
            print(f"  Insert done in {time.time()-t0:.1f}s")
        else:
            print(f"  Cache exists — skipping insert.")

        for threshold in THRESHOLDS:
            print(f"  threshold={threshold:.2f} ...", end=" ", flush=True)
            tp, fp, fn, tn, avg_ms, wall = run_queries(test_pairs, threshold)
            precision, recall, f1 = evaluate_metrics(tp, fp, fn)
            print(f"F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  lat={avg_ms:.1f}ms")

            rows.append({
                "system":    variant_name,
                "embedding": embed_obj.name if hasattr(embed_obj, "name") else variant_name,
                "threshold": threshold,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": round(precision, 4),
                "recall":    round(recall,    4),
                "f1":        round(f1,        4),
                "avg_latency_ms": round(avg_ms, 2),
                "wall_time_s":    round(wall,   2),
            })

    # ── ContextCache variant ──
    print(f"\n{'='*60}")
    print("Variant: ContextCache-MiniLM  (BatchOptimizedEncoder)")
    print(f"{'='*60}")

    cc_cache_dir = os.path.join(CACHE_ROOT, "contextcache_minilm")

    for threshold in THRESHOLDS:
        print(f"  threshold={threshold:.2f} ...", end=" ", flush=True)
        tp, fp, fn, tn, avg_ms, wall = run_contextcache_variant(
            test_pairs, threshold, cc_cache_dir
        )
        precision, recall, f1 = evaluate_metrics(tp, fp, fn)
        print(f"F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  lat={avg_ms:.1f}ms")

        rows.append({
            "system":    "ContextCache-MiniLM",
            "embedding": "all-MiniLM-L6-v2 + BatchOptimizedEncoder",
            "threshold": threshold,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "avg_latency_ms": round(avg_ms, 2),
            "wall_time_s":    round(wall,   2),
        })

    # ── write CSV ──
    fieldnames = ["system", "embedding", "threshold",
                  "TP", "FP", "FN", "TN",
                  "precision", "recall", "f1",
                  "avg_latency_ms", "wall_time_s"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to: {out_csv}")

    # ── print summary table ──
    print(f"\n{'='*95}")
    print(f"{'System':<25} {'Thresh':>7} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Lat(ms)':>9}")
    print("-" * 95)
    for r in rows:
        print(f"{r['system']:<25} {r['threshold']:>7.2f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['TP']:>5} {r['FP']:>5} {r['FN']:>5} {r['TN']:>5} "
              f"{r['avg_latency_ms']:>9.1f}")
    print(f"{'='*95}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Delete all caches and rebuild from scratch")
    parser.add_argument("--out", default="results/ablation.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    main(reset=args.reset, out_csv=args.out)
