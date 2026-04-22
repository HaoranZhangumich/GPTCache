# CSE 584 — Semantic Caching Extensions for GPTCache

This repository extends [GPTCache](https://github.com/zilliztech/GPTCache) with two independent
research contributions for the CSE 584 course project.

| Contribution | Author | Module |
|---|---|---|
| **ContextCache** — context-aware two-stage retrieval | Haoran Zhang | `gptcache/similarity_evaluation/context_match.py` |
| **LLM Intent Embedding + Validator** — LLM-based semantic embedding and hit validation | Hanfu Hou | `gptcache/embedding/llm_intent.py`, `gptcache/similarity_evaluation/llm_validator.py` |

Demo video: https://youtu.be/R3NByaQS7Ws

---

## Repository Layout

```
GPTCache/
├── gptcache/
│   ├── adapter/adapter.py              # ContextCache: two-stage retrieval adapter
│   ├── config.py                       # ContextCache: similarity_threshold + dialuoge_threshold
│   ├── core.py                         # ContextCache: import_data with context tracking
│   ├── embedding/
│   │   └── llm_intent.py               # LLM Intent: GPT-4o intent extraction + SBERT embed
│   ├── manager/scalar_data/
│   │   └── sql_storage.py              # ContextCache: dynamic embed_dim fix
│   └── similarity_evaluation/
│       ├── context_match.py            # ContextCache: BatchOptimizedEncoder evaluation
│       └── llm_validator.py            # LLM Intent: LLM-based cache hit validator
├── benchmark_contextcache.py           # ContextCache benchmark (QQP, 300/1000)
├── ablation_study.py                   # Embedding ablation → results/ablation.csv
├── train_context_model.py              # Train BatchOptimizedEncoder on QQP pairs
├── examples/
│   ├── benchmark/benchmark_llm_intent.py   # LLM Intent benchmark (QQP)
│   └── cse584_llm_intent_demo.py           # LLM Intent interactive demo
├── CSE584_LLM_Intent_Report.md             # LLM Intent detailed report
├── results/
│   └── ablation.csv                    # Pre-computed ablation results
├── README.md                           # This file
└── install.txt                         # Full installation instructions
```

---

## Contribution 1 — ContextCache (Haoran Zhang)

### What it does

Standard GPTCache treats each query independently. ContextCache adds a **two-stage retrieval
architecture** that checks both the current query similarity *and* the conversational context
before returning a cached answer. This prevents false hits when similar questions appear in
different dialogue contexts.

**Two-stage matching (inline in `adapter/adapter.py`):**
1. **Stage 1** — cosine similarity between query embeddings > `similarity_threshold`
2. **Stage 2** — mean cosine similarity between full context sequences > `dialuoge_threshold`

**Trained `BatchOptimizedEncoder`** (multi-layer cross-attention transformer) learns richer
context representations from labeled QQP pairs using triplet margin loss.

### Quick Start

**Install:**
```bash
conda create -n contextcache python=3.11 -y
conda activate contextcache
conda install -c conda-forge faiss-cpu -y
pip install -e .
pip install "sentence-transformers==2.7.0" torch numpy
```

**Dataset** — place `similiar_qqp.json.gz` from the GPTCache repo at:
```
../GPTCache/examples/benchmark/similiar_qqp.json.gz
```

**Run the benchmark** (300 inserted, 1000 tested = 300 pos + 700 neg):
```bash
python benchmark_contextcache.py --threshold 0.75 --reset
```

Expected output:
```
=======================================================
ContextCache Benchmark Results  (threshold=0.75)
=======================================================
Total queries        : 1000
  Positives (in cache): 300
  Negatives (not in)  : 700
Cache hit correct (TP): 244
Cache hit wrong   (FP): 97
Missed positives  (FN): 39
Correct miss      (TN): 620
Precision            : 0.7155
Recall               : 0.8622
F1 Score             : 0.7821
Avg query latency    : ~16ms
```

**Run the full embedding ablation** (4 models × 5 thresholds + ContextCache):
```bash
python ablation_study.py --reset --out results/ablation.csv
```

**Train the context encoder** (optional — requires `similiar_qqp_full.json.gz`):
```bash
python train_context_model.py --epochs 20 --batch_size 128
# Checkpoint saved to results/qqp_train/best_model.pth
```

### Ablation Results

Pre-computed results from `results/ablation.csv`
(dataset: similiar_qqp, 300 inserted, 1000 tested):

| System | Threshold | Precision | Recall | F1 | Avg Latency |
|--------|-----------|-----------|--------|----|-------------|
| GPTCache-ONNX (768d) | 0.70 | 0.7295 | 0.9435 | **0.8228** | 168ms |
| GPTCache-ONNX (768d) | 0.75 | 0.7623 | 0.8728 | 0.8138 | 178ms |
| GPTCache-MiniLM (384d) | 0.70 | 0.6735 | 0.9395 | 0.7845 | 18ms |
| GPTCache-MiniLM (384d) | 0.75 | 0.7155 | 0.8622 | 0.7821 | 16ms |
| GPTCache-BGE-Small (384d) | 0.85 | 0.7403 | 0.8028 | 0.7703 | 22ms |
| GPTCache-BGE-Base (768d) | 0.75 | 0.6374 | 0.9892 | 0.7753 | 25ms |
| ContextCache-MiniLM | 0.70 | 0.6735 | 0.9395 | 0.7845 | 16ms |
| ContextCache-MiniLM | 0.75 | 0.7155 | 0.8622 | **0.7821** | 16ms |

Key findings:
- ONNX achieves highest F1 (0.82) but is **10× slower** than SBERT models
- MiniLM gives the best speed/accuracy trade-off at ~16ms
- ContextCache's two-stage adapter gains further advantage on multi-turn scenarios

---

## Contribution 2 — LLM Intent Embedding + Validator (Hanfu Hou)

See [CSE584_LLM_Intent_Report.md](CSE584_LLM_Intent_Report.md) for the full write-up.

### What it does

Instead of embedding raw query text, **Step 1** calls GPT-4o to extract a structured
"intent signature" (task type, domain, key entities, keywords), then **Step 2** embeds
that signature with a local SBERT model. Two paraphrases of the same question produce
nearly identical vectors regardless of surface wording.

The **LLM Validator** sits after vector search and re-scores each cache hit candidate
by asking the LLM: *"Is this cached answer actually appropriate for the new question?"*
This catches false positives that embedding similarity alone misses (e.g. Python vs Java
sorting questions).

**Pipeline:**
```
Raw query
   ↓ GPT-4o (Vocareum API) — intent extraction
{ task_type, domain, required_facts, constraints, intent_keywords }
   ↓ SBERT local model — embed intent text
384-dim vector → FAISS search → [optional] LLM Validator re-score → cache answer
```

### Quick Start

**Requires a Vocareum API key** (Canvas → CSE 584 → Vocareum LTI → AI sticky note icon).

**Run the benchmark:**
```bash
cd examples/benchmark

# Quick test (20 queries)
python benchmark_llm_intent.py \
    --api-key YOUR_VOC_KEY \
    --base-url https://genai.vocareum.com/v1 \
    --limit 20

# Full test (999 queries, matches teammate's benchmark format)
python benchmark_llm_intent.py \
    --api-key YOUR_VOC_KEY \
    --base-url https://genai.vocareum.com/v1 \
    --limit 999

# Threshold sweep
python benchmark_llm_intent.py \
    --api-key YOUR_VOC_KEY \
    --base-url https://genai.vocareum.com/v1 \
    --limit 100 --sweep
```

**Use in code:**
```python
from gptcache.embedding.llm_intent import LLMIntentEmbedding
from gptcache.similarity_evaluation.llm_validator import LLMValidatorEvaluation

# Step 1: LLM intent embedding
embed = LLMIntentEmbedding(
    openai_api_key="voc-...",
    openai_base_url="https://genai.vocareum.com/v1",
    llm_model="@azure-1/gpt-4o",
)

# Step 2: LLM validator (optional, replaces SearchDistanceEvaluation)
validator = LLMValidatorEvaluation(
    openai_api_key="voc-...",
    openai_base_url="https://genai.vocareum.com/v1",
)

cache.init(
    embedding_func=embed.to_embeddings,
    similarity_evaluation=validator,
    ...
)
```

---

## Installation

See [install.txt](install.txt) for the full step-by-step guide covering both contributions,
including all known issues and fixes.

---

## Bug Fixes in This Fork

| File | Fix |
|------|-----|
| `gptcache/core.py` | `SummarizationContextProcess` is now lazy-loaded (only when `input_summary_len` is set) — prevents segfault on import |
| `gptcache/adapter/adapter.py` | `llm_data["choices"]` access is now guarded against `None` return from cache-miss handler |
| `gptcache/manager/scalar_data/sql_storage.py` | Context data reshape uses dynamic `embed_dim` instead of hardcoded 768 — fixes 384-dim embedding support |
