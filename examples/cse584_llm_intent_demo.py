"""
CSE 584 Project — LLM Intent Embedding Demo & Evaluation
=========================================================

Compares three caching approaches on a paraphrase test set:
  1. Baseline:   ONNX embedding + SearchDistanceEvaluation  (original GPTCache)
  2. Teammate:   OpenAI text-embedding-ada-002 + SearchDistanceEvaluation
  3. Ours:       LLMIntentEmbedding + LLMValidatorEvaluation

How to run:
    1. Set your Vocareum credentials below (or use environment variables)
    2. pip install openai gptcache
    3. python examples/cse584_llm_intent_demo.py

Vocareum credentials are in Canvas → CSE 584 → Vocareum LTI → AI sticky note icon.
"""

import os
import time
import json

# ── Credentials ──────────────────────────────────────────────────────────────
# Fill these in from Canvas Vocareum, or set as environment variables
API_KEY      = os.environ.get("VOCAREUM_API_KEY", "YOUR_VOCAREUM_API_KEY_HERE")
BASE_URL     = os.environ.get("VOCAREUM_BASE_URL", "YOUR_VOCAREUM_BASE_URL_HERE")  # e.g. https://...openai.azure.com/
LLM_MODEL    = "gpt-4o"
EMBED_MODEL  = "text-embedding-ada-002"
# ─────────────────────────────────────────────────────────────────────────────

# Test set: (original_query, paraphrase, should_hit=True/False)
# Each pair tests whether a semantically equivalent paraphrase correctly hits
# the cache populated with the original query.
TEST_PAIRS = [
    # Should hit — same intent, different wording
    ("What is the capital of France?",           "Tell me the capital city of France.",           True),
    ("How do I reverse a string in Python?",     "What's the Python way to flip a string?",       True),
    ("Explain the concept of recursion.",        "Can you describe what recursion means?",         True),
    ("What causes climate change?",              "Why is global warming happening?",               True),
    ("Summarize the French Revolution briefly.", "Give me a short summary of the French Revolution.", True),

    # Should NOT hit — similar surface text but different intent
    ("How do I sort a list in Python?",          "How do I sort a list in Java?",                 False),
    ("What year did World War I start?",         "What year did World War II start?",              False),
    ("What is 15% of 200?",                      "What is 15% of 300?",                            False),
    ("Who wrote Romeo and Juliet?",              "Who directed the 1996 film Romeo + Juliet?",    False),
]

SIMILARITY_THRESHOLD = 0.7


def run_evaluation(approach_name, embedding_func, evaluator, dimension, test_pairs):
    """
    Populates a fresh in-memory cache with original queries, then tests
    paraphrases and reports precision/recall/accuracy.
    """
    import numpy as np
    from gptcache import cache, Config
    from gptcache.manager import get_data_manager, CacheBase, VectorBase
    from gptcache.adapter import openai as cached_openai

    # Fresh cache for each approach
    vector_base = VectorBase("faiss", dimension=dimension)
    data_manager = get_data_manager(CacheBase("sqlite", sql_url="sqlite:///:memory:"), vector_base)
    cache.init(
        embedding_func=embedding_func,
        data_manager=data_manager,
        similarity_evaluation=evaluator,
        config=Config(similarity_threshold=SIMILARITY_THRESHOLD),
    )

    results = []
    populate_times = []
    query_times = []

    print(f"\n{'='*60}")
    print(f"  Approach: {approach_name}")
    print(f"{'='*60}")

    # Step 1: Populate cache with original queries
    for original, _, _ in test_pairs:
        t0 = time.time()
        vec = embedding_func(original)
        data_manager.save(original, f"Cached answer for: {original}", vec)
        populate_times.append(time.time() - t0)

    # Step 2: Query with paraphrases and measure hits
    for original, paraphrase, should_hit in test_pairs:
        t0 = time.time()
        query_vec = embedding_func(paraphrase)
        search_results = data_manager.search(query_vec)
        elapsed = time.time() - t0
        query_times.append(elapsed)

        got_hit = False
        if search_results:
            for result in search_results:
                score = evaluator.evaluation(
                    {"question": paraphrase},
                    {"question": result[1], "answers": []},  # simplified
                )
                if score >= SIMILARITY_THRESHOLD:
                    got_hit = True
                    break

        correct = (got_hit == should_hit)
        status = "✓" if correct else "✗"
        hit_str = "HIT " if got_hit else "MISS"
        print(f"  {status} [{hit_str}] {paraphrase[:55]:<55}  (expected: {'hit' if should_hit else 'miss'})")
        results.append(correct)

    acc = sum(results) / len(results) * 100
    avg_query_ms = sum(query_times) / len(query_times) * 1000
    print(f"\n  Accuracy: {acc:.1f}%  |  Avg query latency: {avg_query_ms:.0f}ms")
    return acc, avg_query_ms


def main():
    print("\n" + "=" * 60)
    print("  CSE 584 — GPTCache LLM Intent Embedding Evaluation")
    print("=" * 60)

    if API_KEY == "YOUR_VOCAREUM_API_KEY_HERE":
        print("\n[!] Please set API_KEY and BASE_URL at the top of this script.")
        print("    Get them from Canvas → CSE 584 → Vocareum LTI → AI icon.\n")
        return

    # ── Approach 1: Baseline ONNX ──────────────────────────────────────────
    try:
        from gptcache.embedding import Onnx
        from gptcache.similarity_evaluation import SearchDistanceEvaluation
        onnx_emb = Onnx()
        run_evaluation(
            "Baseline (ONNX + SearchDistance)",
            onnx_emb.to_embeddings,
            SearchDistanceEvaluation(),
            onnx_emb.dimension,
            TEST_PAIRS,
        )
    except Exception as e:
        print(f"\n[Baseline] Skipped ({e})")

    # ── Approach 2: Teammate — OpenAI raw embedding ────────────────────────
    try:
        from openai import OpenAI as OAI
        import numpy as np
        oai = OAI(api_key=API_KEY, base_url=BASE_URL)

        def openai_embed(text):
            resp = oai.embeddings.create(model=EMBED_MODEL, input=text)
            return np.array(resp.data[0].embedding, dtype=np.float32)

        from gptcache.similarity_evaluation import SearchDistanceEvaluation
        run_evaluation(
            "Teammate (OpenAI text-embedding-ada-002 + SearchDistance)",
            openai_embed,
            SearchDistanceEvaluation(),
            1536,
            TEST_PAIRS,
        )
    except Exception as e:
        print(f"\n[Teammate] Skipped ({e})")

    # ── Approach 3: Ours — LLM Intent + Validator ─────────────────────────
    try:
        from gptcache.embedding import LLMIntentEmbedding
        from gptcache.similarity_evaluation import LLMValidatorEvaluation

        intent_emb = LLMIntentEmbedding(
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            llm_model=LLM_MODEL,
            embedding_model=EMBED_MODEL,
        )
        llm_validator = LLMValidatorEvaluation(
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            llm_model=LLM_MODEL,
        )
        run_evaluation(
            "Ours (LLMIntentEmbedding + LLMValidatorEvaluation)",
            intent_emb.to_embeddings,
            llm_validator,
            intent_emb.dimension,
            TEST_PAIRS,
        )
    except Exception as e:
        print(f"\n[Ours] Failed ({e})")
        raise

    print("\n" + "=" * 60)
    print("  Done. See results above for comparison.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
