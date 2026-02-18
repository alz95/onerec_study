# OneRec 2502.18965 Reproduction Scaffold (PyTorch)

This repository now contains a **minimal, runnable PyTorch scaffold** for reproducing
OneRec (arXiv:2502.18965) in a disciplined, iteration-friendly way.

## What is included

- Structured paper-to-code plan in `docs/spec_2502.md`
- Minimal sequential recommendation training loop (BPR objective)
- Config-driven entrypoint
- Core ranking metrics: Recall@K and NDCG@K
- Synthetic data path so the pipeline can run end-to-end immediately

> Note: this is a starting point focused on engineering reproducibility. Exact paper-level
alignment still requires dataset/protocol implementation from the paper.

## Quick start

```bash
python train.py --config configs/onerec_2502_minimal.yaml
```

## Project structure

- `configs/` - experiment configs
- `docs/` - structured reproduction specification
- `docs/results/` - experiment result reports
- `onerec/` - package modules
  - `data/` - dataset and dataloader helpers
  - `models/` - model definitions
  - `training/` - trainer logic
  - `metrics/` - ranking metrics

## Next steps

1. Implement exact dataset split protocol from 2502.18965.
2. Replace synthetic dataset path with real benchmark pipeline.
3. Add paper-specific model blocks and losses as ablations.
4. Run multi-seed experiments and compare against reported metrics.

## Results
- ML-1M (full, all-item eval, MPS): [2026-02-18-ml1m](docs/results/2026-02-18-ml1m.md)

## Dashboard
- Start the dashboard server:
  - `. .venv/bin/activate`
  - `python dashboard.py`
- Open http://localhost:8765/ to view loss/time/Recall@K/NDCG@K


## Dataset download links

If you want to manually download paper-related recommendation datasets first, see:

- `docs/dataset_links.md`
