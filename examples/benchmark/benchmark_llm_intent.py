"""
CSE 584 — LLM Intent Embedding Benchmark
==========================================
Matches the same output format as the teammate's benchmark_embedding_comparison.py
Uses the same similiar_qqp.json dataset and reports the same metrics.

Usage:
    # Quick test (20 queries, cheap):
    python benchmark_llm_intent.py --api-key YOUR_KEY --base-url YOUR_URL --limit 20

    # Full test (999 queries, matches teammate):
    python benchmark_llm_intent.py --api-key YOUR_KEY --base-url YOUR_URL --limit 999

    # Threshold sweep:
    python benchmark_llm_intent.py --api-key YOUR_KEY --base-url YOUR_URL --limit 100 --sweep

Get your API key and base URL from:
    Canvas → CSE 584 → Vocareum LTI → click the AI sticky note icon (lower left)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # fix segfault on macOS with PyTorch

import argparse
import gzip
import io
import json
import sys
import tarfile
import time

import numpy as np

# ── make sure gptcache package is importable ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── load dataset ─────────────────────────────────────────────────────────────

def load_qqp_data(limit: int = 999):
    """Load origin/similar pairs from similiar_qqp.json.gz (tar inside gz)."""
    gz_path = os.path.join(os.path.dirname(__file__), "similiar_qqp.json.gz")
    with gzip.open(gz_path, "rb") as gz:
        raw = gz.read()
    tar = tarfile.open(fileobj=io.BytesIO(raw))
    member = tar.getmembers()[0]
    content = tar.extractfile(member).read().decode("utf-8")
    data = json.loads(content)
    return data[:limit]


# ── core benchmark runner ─────────────────────────────────────────────────────

def run_benchmark(label, embed_fn, dimension, threshold, data, use_llm_validator=False,
                  api_key=None, base_url=None, llm_model="gpt-4o"):
    """
    Populate a fresh FAISS+SQLite cache with 'origin' queries,
    then query with 'similar' queries and measure hit rate / precision.

    Returns dict with all metrics.
    """
    from gptcache import cache, Config
    from gptcache.manager import get_data_manager, CacheBase, VectorBase

    print(f"\n{'='*70}")
    print(f"  LLM Intent Benchmark: {label}")
    print(f"  similarity_threshold={threshold}")
    print(f"{'='*70}")

    # Fresh in-memory cache each run
    vector_base = VectorBase("faiss", dimension=dimension)
    cache_base = CacheBase("sqlite", sql_url="sqlite:///:memory:")
    data_manager = get_data_manager(cache_base, vector_base, max_size=200000)

    if use_llm_validator:
        from gptcache.similarity_evaluation.llm_validator import LLMValidatorEvaluation
        evaluator = LLMValidatorEvaluation(
            openai_api_key=api_key,
            openai_base_url=base_url,
            llm_model=llm_model,
        )
    else:
        from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
        evaluator = SearchDistanceEvaluation()

    cache.init(
        embedding_func=embed_fn,
        data_manager=data_manager,
        similarity_evaluation=evaluator,
        config=Config(similarity_threshold=threshold),
    )

    # ── Step 1: Insert origin questions ──────────────────────────────────────
    print("Inserting data...")
    t0 = time.time()
    origins = [pair["origin"] for pair in data]
    answers = [str(i) for i in range(len(data))]       # answer = index string
    cache.import_data(questions=origins, answers=answers)
    insert_time = time.time() - t0
    print(f"Insert done in {insert_time:.2f}s")

    # ── Step 2: Query with similar questions ──────────────────────────────────
    print("Running queries...")

    hit_positive = 0    # cache hit, returned correct answer
    hit_negative = 0    # cache hit, returned wrong answer
    miss_count = 0      # no cache hit (None returned)
    fail_count = 0      # exception
    hit_latencies = []
    embed_times = []
    search_times = []

    for i, pair in enumerate(data):
        try:
            t_start = time.time()

            # Step 1: embed the similar query
            t_embed = time.time()
            query_vec = embed_fn(pair["similar"])
            embed_elapsed = time.time() - t_embed
            embed_times.append(embed_elapsed)

            # Step 2: vector search → returns [(score, vector_id), ...] or None
            t_search = time.time()
            search_results = data_manager.search(query_vec, top_k=1)
            search_elapsed = time.time() - t_search
            search_times.append(search_elapsed)

            elapsed = time.time() - t_start

            if not search_results:
                miss_count += 1
                continue

            # Step 3: fetch the actual CacheData object using the search result
            top_result = search_results[0]
            cache_data = data_manager.get_scalar_data(top_result, extra_param={})

            if cache_data is None:
                miss_count += 1
                continue

            # Step 4: evaluate similarity with proper src/cache dicts
            src_dict = {
                "question": pair["similar"],
                "embedding": query_vec,
            }
            cache_dict = {
                "question": cache_data.question,
                "answers": cache_data.answers,
                "embedding": cache_data.embedding_data,
                "search_result": top_result,
            }
            score = evaluator.evaluation(src_dict, cache_dict)
            eval_min, eval_max = evaluator.range()
            normalized = (score - eval_min) / (eval_max - eval_min + 1e-9)

            if normalized >= threshold:
                hit_latencies.append(elapsed * 1000)  # ms
                # correct hit if the cached answer index matches this pair's index
                cached_answer = cache_data.answers[0].answer if cache_data.answers else ""
                if cached_answer == str(i):
                    hit_positive += 1
                else:
                    hit_negative += 1
            else:
                miss_count += 1

        except Exception as e:
            fail_count += 1
            print(f"  [query {i}] ERROR: {e}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    total = len(data)
    total_hits = hit_positive + hit_negative
    precision = (hit_positive / total_hits * 100) if total_hits > 0 else 0.0
    avg_hit_latency = (sum(hit_latencies) / len(hit_latencies)) if hit_latencies else 0.0
    total_query_time = sum(embed_times) + sum(search_times)
    avg_embed = (sum(embed_times) / len(embed_times)) if embed_times else 0.0
    avg_search = (sum(search_times) / len(search_times)) if search_times else 0.0

    print(f"\n--- Results: {label} ---")
    print(f"Total queries      : {total}")
    print(f"Cache hit positive : {hit_positive}  ({hit_positive/total*100:.1f}%)")
    print(f"Cache hit negative : {hit_negative}  ({hit_negative/total*100:.1f}%)")
    print(f"Cache miss (None)  : {miss_count}  ({miss_count/total*100:.1f}%)")
    print(f"Failures           : {fail_count}")
    print(f"Precision          : {precision:.1f}%")
    print(f"Avg hit latency    : {avg_hit_latency:.1f}ms")
    print(f"Total query time   : {total_query_time:.2f}s")
    print(f"Avg embed time     : {avg_embed:.4f}")
    print(f"Avg search time    : {avg_search:.4f}")

    return {
        "label": label,
        "threshold": threshold,
        "total": total,
        "hit_positive": hit_positive,
        "hit_negative": hit_negative,
        "miss": miss_count,
        "failures": fail_count,
        "precision": precision,
        "avg_hit_latency_ms": avg_hit_latency,
        "total_query_time": total_query_time,
        "avg_embed": avg_embed,
        "avg_search": avg_search,
    }


def threshold_sweep(label, embed_fn, dimension, data, thresholds,
                    use_llm_validator=False, api_key=None, base_url=None, llm_model="gpt-4o"):
    """Run the same benchmark across multiple thresholds and print a summary table."""
    rows = []
    for t in thresholds:
        r = run_benchmark(label, embed_fn, dimension, t, data,
                          use_llm_validator=use_llm_validator,
                          api_key=api_key, base_url=base_url, llm_model=llm_model)
        rows.append(r)

    print(f"\n{'='*70}")
    print(f"THRESHOLD SWEEP SUMMARY: {label}")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Hit+':<8} {'Hit-':<8} {'Miss':<8} {'Avg(ms)'}")
    print("-" * 60)
    for r in rows:
        print(f"{r['threshold']:<12.2f} {r['precision']:<11.1f}% "
              f"{r['hit_positive']:<8} {r['hit_negative']:<8} "
              f"{r['miss']:<8} {r['avg_hit_latency_ms']:.1f}")


# ── main ──────────────────────────────────────────────────────────────────────

API_KEY  = "voc-734807076204245434089369b8ae6532d552.80740985"
BASE_URL = "https://genai.vocareum.com/v1"
MODEL    = "@azure-1/gpt-4o"


def main():
    parser = argparse.ArgumentParser(description="LLM Intent Embedding Benchmark for CSE 584")
    parser.add_argument("--api-key",  default=API_KEY,  help="Vocareum API key")
    parser.add_argument("--base-url", default=BASE_URL, help="Vocareum base URL")
    parser.add_argument("--model",    default=MODEL,    help="LLM model")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2",
                        help="Local SBERT model for embedding (default: all-MiniLM-L6-v2, free/local)")
    parser.add_argument("--limit",    type=int, default=50,
                        help="Number of query pairs to test (default: 50, use 999 to match teammate)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity threshold (default: 0.7)")
    parser.add_argument("--sweep",    action="store_true",
                        help="Run threshold sweep like teammate's script")
    parser.add_argument("--with-validator", action="store_true",
                        help="Also run LLMIntent + LLMValidator (costs more API calls)")
    args = parser.parse_args()

    print(f"\nLoading dataset (first {args.limit} pairs)...")
    data = load_qqp_data(limit=args.limit)
    print(f"Loaded {len(data)} pairs.")

    # Build the LLM intent embedding function
    from gptcache.embedding.llm_intent import LLMIntentEmbedding
    intent_embedder = LLMIntentEmbedding(
        openai_api_key=args.api_key,
        openai_base_url=args.base_url,
        llm_model=args.model,
        sbert_model=args.sbert_model,
    )

    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]

    if args.sweep:
        # ── Threshold sweep: LLMIntent + SearchDistance ───────────────────────
        threshold_sweep(
            label="llm_intent",
            embed_fn=intent_embedder.to_embeddings,
            dimension=intent_embedder.dimension,
            data=data,
            thresholds=thresholds,
            use_llm_validator=False,
            api_key=args.api_key,
            base_url=args.base_url,
            llm_model=args.model,
        )

        if args.with_validator:
            # ── Threshold sweep: LLMIntent + LLMValidator ─────────────────────
            threshold_sweep(
                label="llm_intent+validator",
                embed_fn=intent_embedder.to_embeddings,
                dimension=intent_embedder.dimension,
                data=data,
                thresholds=thresholds,
                use_llm_validator=True,
                api_key=args.api_key,
                base_url=args.base_url,
                llm_model=args.model,
            )
    else:
        # ── Single threshold run ──────────────────────────────────────────────
        run_benchmark(
            label="llm_intent",
            embed_fn=intent_embedder.to_embeddings,
            dimension=intent_embedder.dimension,
            threshold=args.threshold,
            data=data,
            use_llm_validator=False,
            api_key=args.api_key,
            base_url=args.base_url,
            llm_model=args.model,
        )

        if args.with_validator:
            run_benchmark(
                label="llm_intent+validator",
                embed_fn=intent_embedder.to_embeddings,
                dimension=intent_embedder.dimension,
                threshold=args.threshold,
                data=data,
                use_llm_validator=True,
                api_key=args.api_key,
                base_url=args.base_url,
                llm_model=args.model,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
