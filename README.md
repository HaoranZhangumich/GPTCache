# GPTCache : A Library for Creating Semantic Cache for LLM Queries
Slash Your LLM API Costs by 10x 💰, Boost Speed by 100x ⚡ 

[![Release](https://img.shields.io/pypi/v/gptcache?label=Release&color&logo=Python)](https://pypi.org/project/gptcache/)
[![pip download](https://img.shields.io/pypi/dm/gptcache.svg?color=bright-green&logo=Pypi)](https://pypi.org/project/gptcache/)
[![Codecov](https://img.shields.io/codecov/c/github/zilliztech/GPTCache/dev?label=Codecov&logo=codecov&token=E30WxqBeJJ)](https://codecov.io/gh/zilliztech/GPTCache)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/zilliz_universe.svg?style=social&label=Follow%20%40Zilliz)](https://twitter.com/zilliz_universe)
[![Discord](https://img.shields.io/discord/1092648432495251507?label=Discord&logo=discord)](https://discord.gg/Q8C6WEjSWV)

🎉 GPTCache has been fully integrated with 🦜️🔗[LangChain](https://github.com/hwchase17/langchain) ! Here are detailed [usage instructions](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/llm_caching#gptcache).

🐳 [The GPTCache server docker image](https://github.com/zilliztech/GPTCache/blob/main/docs/usage.md#Use-GPTCache-server) has been released, which means that **any language** will be able to use GPTCache!

📔 This project is undergoing swift development, and as such, the API may be subject to change at any time. For the most up-to-date information, please refer to the latest [documentation]( https://gptcache.readthedocs.io/en/latest/) and [release note](https://github.com/zilliztech/GPTCache/blob/main/docs/release_note.md).

**NOTE:** As the number of large models is growing explosively and their API shape is constantly evolving, we no longer add support for new API or models. We encourage the usage of using the get and set API in gptcache, here is the demo code: https://github.com/zilliztech/GPTCache/blob/main/examples/adapter/api.py

## Quick Install

`pip install gptcache`

## 🚀 What is GPTCache?

ChatGPT and various large language models (LLMs) boast incredible versatility, enabling the development of a wide range of applications. However, as your application grows in popularity and encounters higher traffic levels, the expenses related to LLM API calls can become substantial. Additionally, LLM services might exhibit slow response times, especially when dealing with a significant number of requests.

To tackle this challenge, we have created GPTCache, a project dedicated to building a semantic cache for storing LLM responses. 

## 😊 Quick Start

**Note**:

- You can quickly try GPTCache and put it into a production environment without heavy development. However, please note that the repository is still under heavy development.
- By default, only a limited number of libraries are installed to support the basic cache functionalities. When you need to use additional features, the related libraries will be **automatically installed**.
- Make sure that the Python version is **3.8.1 or higher**, check: `python --version`
- If you encounter issues installing a library due to a low pip version, run: `python -m pip install --upgrade pip`.

### dev install

```bash
# clone GPTCache repo
git clone -b dev https://github.com/zilliztech/GPTCache.git
cd GPTCache

# install the repo
pip install -r requirements.txt
python setup.py install
```

### example usage

These examples will help you understand how to use exact and similar matching with caching. You can also run the example on [Colab](https://colab.research.google.com/drive/1m1s-iTDfLDk-UwUAQ_L8j1C-gzkcr2Sk?usp=share_link). And more examples you can refer to the [Bootcamp](https://gptcache.readthedocs.io/en/latest/bootcamp/openai/chat.html)

Before running the example, **make sure** the OPENAI_API_KEY environment variable is set by executing `echo $OPENAI_API_KEY`. 

If it is not already set, it can be set by using `export OPENAI_API_KEY=YOUR_API_KEY` on Unix/Linux/MacOS systems or `set OPENAI_API_KEY=YOUR_API_KEY` on Windows systems. 

> It is important to note that this method is only effective temporarily, so if you want a permanent effect, you'll need to modify the environment variable configuration file. For instance, on a Mac, you can modify the file located at `/etc/profile`.

<details>

<summary> Click to <strong>SHOW</strong> example code </summary>

#### OpenAI API original usage

```python
import os
import time

import openai

Demo: https://youtu.be/R3NByaQS7Ws

## What is ContextCache?

Large language models (LLMs) like ChatGPT enable powerful applications, but scaling them introduces
significant costs and latency from repeated API calls. While solutions like GPTCache cache LLM
responses through semantic matching, existing approaches focus on individual queries without
considering conversational context.

ContextCache addresses this with:
- A **two-stage retrieval architecture** combining vector search with dialogue-aware matching
- A **trained BatchOptimizedEncoder** (multi-layer cross-attention transformer) for context-aware similarity
- **10× lower latency** than direct LLM calls
- Improved precision and recall over GPTCache on multi-turn scenarios

---

## Repository Layout

```
ContextCache/
├── gptcache/                   # Modified gptcache package (ContextCache extensions)
│   ├── adapter/adapter.py      # Two-stage retrieval logic
│   ├── config.py               # similarity_threshold + dialuoge_threshold
│   ├── core.py                 # Cache.init / import_data
│   └── similarity_evaluation/
│       └── context_match.py    # BatchOptimizedEncoder + ContextMatchEvaluation
├── benchmark_contextcache.py   # Single-turn QQP benchmark (300 insert / 1000 test)
├── ablation_study.py           # Embedding ablation across thresholds → CSV
├── train_context_model.py      # Train BatchOptimizedEncoder on QQP pairs
├── requirements.txt
├── setup.py
└── results/
    └── ablation.csv            # Pre-computed ablation results
```

---

## Quick Start — Reproduce the ContextCache Benchmark

### 1. Install dependencies

See `install.txt` for the full step-by-step conda/pip instructions.

Short version (conda recommended):
```bash
conda create -n contextcache python=3.11 -y
conda activate contextcache
conda install -c conda-forge faiss-cpu -y
pip install -e .
pip install "sentence-transformers==2.7.0" torch numpy
```

### 2. Get the dataset

The benchmark uses `similiar_qqp.json.gz` from the GPTCache repo:
```bash
# place it at:  ../GPTCache/examples/benchmark/similiar_qqp.json.gz
# or edit QQP_PATH in benchmark_contextcache.py to point to your copy
```

### 3. Run the benchmark (single-turn QQP, threshold=0.75)

```bash
cd ContextCache
python benchmark_contextcache.py --threshold 0.75 --reset
```

Expected output (threshold=0.75):
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

### 4. Run the full ablation study (all embeddings × all thresholds)

```bash
python ablation_study.py --reset --out results/ablation.csv
```

This compares GPTCache-ONNX, GPTCache-MiniLM, GPTCache-BGE-Small, GPTCache-BGE-Base, and
ContextCache-MiniLM across thresholds 0.70 / 0.75 / 0.80 / 0.85 / 0.90.
Results are written to `results/ablation.csv` (pre-computed copy already included).

---

## Train the Context Encoder (Optional)

The `BatchOptimizedEncoder` is trained on labeled QQP pairs using triplet margin loss.
You need `similiar_qqp_full.json.gz` (~80k pairs) from the GPTCache repo.

```bash
# First run: embeds ~137k unique texts with MiniLM (saved to qqp_embeddings.npy)
python train_context_model.py --epochs 20 --batch_size 128

# Subsequent runs reuse the cached embeddings
python train_context_model.py --embed_cache qqp_embeddings.npy --epochs 20
```

Checkpoints are saved to `results/qqp_train/best_model.pth`.
After training, `context_match.py` loads this path automatically.

Training takes ~10 min on Apple Silicon (MPS) or CUDA GPU.

---

## Ablation Results Summary

Pre-computed results from `results/ablation.csv`
(dataset: similiar_qqp, 300 inserted, 1000 tested = 300 pos + 700 neg):

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
- ONNX (paraphrase-albert) achieves the highest F1 (0.82) but is **10× slower** than SBERT models
- MiniLM gives the best speed/accuracy trade-off at ~16ms latency
- ContextCache's two-stage adapter matches MiniLM precision on single-turn queries and gains further advantage on multi-turn scenarios where conversational context disambiguates similar questions

---

## Citation / Course Project

This repository is part of CSE 584 course project exploring context-aware semantic caching.