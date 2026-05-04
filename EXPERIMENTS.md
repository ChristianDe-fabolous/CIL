# CIL Monocular Depth Estimation — Idea & Experiment Plan

## Problem

Predict pixel-wise depth from a single RGB image.  
Metric: **scale-invariant RMSE** (silog) — penalises structural depth errors, not global scale drift.

---

## Core Idea: In-Context Depth Estimation

Standard approach: ViT encoder → decoder → depth map.  
Our bet: give the model **K reference (image, depth) pairs** at inference time so it can adapt its prediction to the scene geometry of similar examples — without retraining.

### Architecture

```
query_img ──► ViT encoder ──► query tokens (B, N, D)
                                        │
ctx_imgs  ──► ViT encoder ──► img tokens (B, K*N, D)
ctx_depths ──► DepthPatcher ─► depth tokens (B, K*N, D)
                                img + depth tokens = context tokens
                                        │
                         num_cross_blocks × CrossAttention
                         (query attends to context)
                                        │
                         num_self_blocks × SelfAttention
                                        │
                              Linear head → depth (B,1,H,W)
```

**Key design choices:**
- Shared ViT encoder for query and context images (no extra parameters, leverages pretrained features)
- Context depth injected as additive patch embeddings (`AvgPool → Linear`) on top of image tokens
- Differential LR: encoder LR = `lr / 50` to stay near pretrained weights; cross/self-attention blocks learn faster
- SILog loss matches the evaluation metric directly

### Retrieval Strategies

Context quality matters. Four strategies, increasing in sophistication:

| Strategy | How | Cost |
|---|---|---|
| `random` | Random K from training set | none |
| `patch_sim` | Cosine sim of mean-pooled ViT patch embeddings | offline index |
| `gt_depth_sim` | Cosine sim of depth log-histograms (64 bins) | offline graph |
| `learned` | Cosine sim of task-encoder features from fine-tuned baseline | requires baseline first |

---

## Experiments

### E0 — Baseline (DepthModel, no context)

Train standard ViT encoder + Transformer decoder.  
Purpose: anchor SI-RMSE number; encoder checkpoint used by `learned` retriever later.

```
python src/train.py --config configs/config.yaml
```

**Variants:**
- E0a: transformer decoder (default)
- E0b: conv decoder (`decoder_type: conv`) — quick sanity that transformer decoder is worth the cost

---

### E1 — In-Context + Random Retrieval

Does adding K random (image, depth) pairs help vs. baseline?  
If yes → cross-attention mechanism works. If no → retrieval quality is all that matters.

```
python src/train_in_context.py --config configs/config_in_context.yaml --strategy random
```

---

### E2 — In-Context + Patch-Sim Retrieval

Replace random with semantically similar context (generic ViT features).

```
# Build index first (once):
python -m src.retrieval.build_index patch_sim --save_path data/retrieval/patch_sim.pt

python src/train_in_context.py --config configs/config_in_context.yaml --strategy patch_sim
```

Expected: better than E1 — similar scenes share depth structure.

---

### E3 — In-Context + GT-Depth-Sim Retrieval

Context selected by depth histogram similarity — near-oracle retrieval quality.

```
# Build graph first (once):
python -m src.retrieval.build_index gt_depth --save_path data/retrieval/gt_depth_graph.pt

python src/train_in_context.py --config configs/config_in_context.yaml --strategy gt_depth_sim
```

Expected: upper bound for retrieval-based gain.

---

### E4 — In-Context + Learned Retrieval

Use fine-tuned task-encoder features (from E0 checkpoint) as retrieval signal.  
Captures what the model learned is structurally similar, not just generic visual similarity.

```
# Build learned graph (requires E0 best.pth):
python -m src.retrieval.build_index learned \
    --save_path data/retrieval/learned_graph.pt \
    --checkpoint checkpoints/vit_depth/best.pth

python src/train_in_context.py --config configs/config_in_context.yaml --strategy learned
```

---

### E5 — Ablation: Number of Context Pairs K

Fix strategy = `patch_sim`, vary K ∈ {1, 2, 4, 8}.  
Tradeoff: more context = more signal but more GPU memory and slower forward pass.

---

### E6 — Ablation: Cross-Attention Depth

Fix strategy = `patch_sim`, K=4, vary `num_cross_blocks` ∈ {1, 3, 5}.  
How many cross-attention layers are needed to extract useful context?

---

## Summary Table

| Exp | Model | Retrieval | Purpose |
|-----|-------|-----------|---------|
| E0a | Baseline (transformer dec) | — | Anchor SI-RMSE |
| E0b | Baseline (conv dec) | — | Decoder ablation |
| E1  | InContext | random | Does context help at all? |
| E2  | InContext | patch_sim | Generic semantic retrieval |
| E3  | InContext | gt_depth_sim | Oracle retrieval upper bound |
| E4  | InContext | learned | Task-specific retrieval |
| E5  | InContext | patch_sim | K ablation (1/2/4/8) |
| E6  | InContext | patch_sim | Cross-block depth ablation |

**Recommended order:** E0a → E1 → E2 → E3 → E0b/E4/E5/E6 (E4 needs E0a checkpoint).
