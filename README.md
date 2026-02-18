# OneRec 2502.18965 Reproduction Scaffold (PyTorch)

This repository now provides an **OpenOneRec-like baseline** on MovieLens-1M so that
method and metric comparisons against classical baselines are done under a fixed protocol.

## What is included

- MovieLens-1M leave-one-out split (timestamp-based)
- OpenOneRec-like encoder-decoder training path
- Sparse top-1 MoE feed-forward blocks in decoder
- Objectives: generative CE / BPR / DPO-style pair preference
- Evaluation modes:
  - sampled ranking (1 positive + N negatives)
  - all-item ranking (with optional seen-item filtering)
- Metrics: HR@K / Recall@K / NDCG@K
- Device support: CUDA > MPS > CPU

## Quick start

1) Download and unzip `ml-1m` so this file exists:
- `data/raw/ml-1m/ratings.dat`

2) Run training:

```bash
python train.py --config configs/onerec_2502_minimal.yaml
```

## Method alignment notes

This code aligns to OpenOneRec at the **method level** by introducing:
- encoder-decoder generation path
- sparse MoE routing in decoder blocks
- optional preference alignment objective (DPO-style pair loss)

It is still a compact reproduction baseline; exact paper parity requires
matching their full tokenizer/indexing, sampling schedule, and reward model pipeline.

## Config knobs for comparable experiments

In `configs/onerec_2502_minimal.yaml`:
- `train.eval_mode`: `sampled` or `all`
- `train.filter_seen_in_eval`: enable seen-item filtering in all-item mode
- `train.objective`: `generative_ce`, `bpr`, or `dpo`
- `train.negatives_per_user`: negatives per user for sampled evaluation

## Project structure

- `configs/` - experiment configs
- `docs/` - reproduction specs and dataset links
- `onerec/` - package modules
  - `data/` - data loaders / splits
  - `models/` - model definitions
  - `training/` - train and eval loops
  - `metrics/` - ranking metrics

## Dataset download links

- see `docs/dataset_links.md`
