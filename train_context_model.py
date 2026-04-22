"""
Training script for ContextCache BatchOptimizedEncoder.

Uses similiar_qqp_full.json.gz (80,854 labeled pairs) to train the model.

Steps:
  1. Load QQP pairs (label=1 similar, label=0 dissimilar)
  2. Pre-compute ONNX embeddings once (saved to cache for reuse)
  3. Build triplets: (anchor, positive, negative)
  4. Train BatchOptimizedEncoder with triplet margin loss
  5. Save best checkpoint

Usage:
    python train_context_model.py
    python train_context_model.py --epochs 30 --batch_size 64
    python train_context_model.py --embed_cache embeddings.npy  # skip re-embedding
"""

import argparse
import json
import os
import random
import tarfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── paths ──────────────────────────────────────────────────────────────────────
QQP_FULL_PATH = "../GPTCache/examples/benchmark/similiar_qqp_full.json.gz"
CHECKPOINT_DIR = "results/qqp_train"
EMBED_CACHE_PATH = "qqp_embeddings.npy"
EMBED_TEXTS_PATH = "qqp_texts.json"

# ── model (copied from context_match.py) ───────────────────────────────────────
from gptcache.similarity_evaluation.context_match import BatchOptimizedEncoder


# ── dataset ────────────────────────────────────────────────────────────────────
class TripletDataset(Dataset):
    """
    Each item is a triplet (anchor_emb, positive_emb, negative_emb).
    All embeddings are shape (1, embed_dim) — single-turn, seq_len=1.
    """

    def __init__(self, embeddings: np.ndarray, pos_pairs, neg_pairs, num_negatives=1):
        self.embeddings = embeddings
        # Build index lookup: text_a → list of positive text_b indices
        self.triplets = []
        neg_b_indices = [b for _, b in neg_pairs]

        for a_idx, b_idx in pos_pairs:
            # Sample random hard negatives from the negative pool
            neg_samples = random.sample(neg_b_indices, min(num_negatives, len(neg_b_indices)))
            for neg_idx in neg_samples:
                self.triplets.append((a_idx, b_idx, neg_idx))

        print(f"Dataset: {len(self.triplets)} triplets from "
              f"{len(pos_pairs)} positive pairs")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_idx, p_idx, n_idx = self.triplets[idx]
        anchor   = torch.tensor(self.embeddings[a_idx], dtype=torch.float32).unsqueeze(0)
        positive = torch.tensor(self.embeddings[p_idx], dtype=torch.float32).unsqueeze(0)
        negative = torch.tensor(self.embeddings[n_idx], dtype=torch.float32).unsqueeze(0)
        return anchor, positive, negative


# ── embedding ──────────────────────────────────────────────────────────────────
def embed_all_texts(texts, batch_size=256):
    """Embed all texts using SBERT all-MiniLM-L6-v2 (fast batch inference)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    total = len(texts)
    embeddings = []
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(embs)
        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {min(i+batch_size, total)}/{total}...")
    return np.concatenate(embeddings, axis=0).astype(np.float32)


# ── training ───────────────────────────────────────────────────────────────────
def train(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # ── load QQP data ──
    print("Loading QQP data...")
    qqp_path = args.qqp_path
    with open(qqp_path, "rb") as f:
        t = tarfile.open(fileobj=f)
        raw = json.loads(t.extractfile(t.getmembers()[0]).read())
    print(f"  Total pairs: {len(raw)}")

    pos_pairs_raw = [(d["text_a"], d["text_b"]) for d in raw if d["label"] == 1]
    neg_pairs_raw = [(d["text_a"], d["text_b"]) for d in raw if d["label"] == 0]
    print(f"  Positive pairs: {len(pos_pairs_raw)}")
    print(f"  Negative pairs: {len(neg_pairs_raw)}")

    # ── build text index ──
    embed_cache = args.embed_cache
    if os.path.isfile(embed_cache) and os.path.isfile(EMBED_TEXTS_PATH):
        print(f"Loading cached embeddings from {embed_cache}...")
        embeddings = np.load(embed_cache)
        with open(EMBED_TEXTS_PATH) as f:
            text2idx = json.load(f)
    else:
        # Collect all unique texts
        all_texts_set = set()
        for a, b in pos_pairs_raw:
            all_texts_set.add(a); all_texts_set.add(b)
        for a, b in neg_pairs_raw:
            all_texts_set.add(a); all_texts_set.add(b)
        all_texts = list(all_texts_set)
        text2idx = {t: i for i, t in enumerate(all_texts)}

        print(f"Embedding {len(all_texts)} unique texts...")
        embeddings = embed_all_texts(all_texts)
        np.save(embed_cache, embeddings)
        with open(EMBED_TEXTS_PATH, "w") as f:
            json.dump(text2idx, f)
        print(f"Saved embeddings to {embed_cache}")

    embed_dim = embeddings.shape[1]
    print(f"Embedding dim: {embed_dim}, total texts: {len(embeddings)}")

    # ── build index pairs ──
    pos_pairs = [(text2idx[a], text2idx[b]) for a, b in pos_pairs_raw
                 if a in text2idx and b in text2idx]
    neg_pairs = [(text2idx[a], text2idx[b]) for a, b in neg_pairs_raw
                 if a in text2idx and b in text2idx]

    # train/val split
    random.shuffle(pos_pairs)
    split = int(len(pos_pairs) * 0.9)
    train_pos, val_pos = pos_pairs[:split], pos_pairs[split:]

    train_dataset = TripletDataset(embeddings, train_pos, neg_pairs)
    val_dataset   = TripletDataset(embeddings, val_pos,   neg_pairs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── model ──
    # num_heads must divide embed_dim; MiniLM=384 → 8 heads (384/8=48 ok)
    model = BatchOptimizedEncoder(
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=3,
        dropout_rate=0.2,
        normalize_output=True,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=args.margin,
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for anchor, positive, negative in train_loader:
            anchor   = anchor.to(device)    # [B, 1, dim]
            positive = positive.to(device)
            negative = negative.to(device)

            # mask: all False (no padding, seq_len=1)
            mask = torch.zeros(anchor.size(0), 1, dtype=torch.bool, device=device)

            a_repr = model(anchor,   mask)
            p_repr = model(positive, mask)
            n_repr = model(negative, mask)

            loss = triplet_loss(a_repr, p_repr, n_repr)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor   = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                mask = torch.zeros(anchor.size(0), 1, dtype=torch.bool, device=device)

                a_repr = model(anchor,   mask)
                p_repr = model(positive, mask)
                n_repr = model(negative, mask)

                loss = triplet_loss(a_repr, p_repr, n_repr)
                val_loss += loss.item()

                # accuracy: positive should be closer than negative
                d_pos = 1 - F.cosine_similarity(a_repr, p_repr)
                d_neg = 1 - F.cosine_similarity(a_repr, n_repr)
                correct += (d_pos < d_neg).sum().item()
                total   += anchor.size(0)

        val_loss /= len(val_loader)
        val_acc   = correct / total * 100
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"acc={val_acc:.1f}% | {elapsed:.0f}s")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc})

        # save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc}, ckpt_path)
            print(f"  Saved {ckpt_path}")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc}, best_path)
            print(f"  *** New best model (val_loss={val_loss:.4f}) ***")

    # save history
    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR}/best_model.pth")
    print(f"\nTo use this model in ContextCache, update context_match.py line 782:")
    print(f'  model_path = "{os.path.abspath(CHECKPOINT_DIR)}/best_model.pth"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qqp_path", default=QQP_FULL_PATH,
                        help="Path to similiar_qqp_full.json.gz")
    parser.add_argument("--embed_cache", default=EMBED_CACHE_PATH,
                        help="Path to save/load pre-computed embeddings (.npy)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Triplet loss margin (default: 0.3)")
    args = parser.parse_args()
    train(args)
