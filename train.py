from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml

from onerec.models.sequential import SimpleSequentialRec
from onerec.training.trainer import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OneRec 2502 minimal PyTorch scaffold")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "mps":
        device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg.get("data", {})
    data_source = data_cfg.get("source", "synthetic")

    if data_source == "ml1m":
        from onerec.data.ml1m import load_ml1m
        ml1m_dir = Path(data_cfg["ml1m_raw_dir"])
        data = load_ml1m(ml1m_dir, rating_threshold=data_cfg.get("rating_threshold", 0), min_seq_len=3)
        model_num_items = data.num_items
    else:
        model_num_items = model_cfg["num_items"]

    model = SimpleSequentialRec(
        num_items=model_num_items,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
    )

    trainer = Trainer(
        model=model,
        cfg=TrainConfig(
            seed=cfg["seed"],
            num_items=model_cfg["num_items"],
            max_seq_len=model_cfg["max_seq_len"],
            batch_size=train_cfg["batch_size"],
            epochs=train_cfg["epochs"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            num_train_samples=train_cfg["num_train_samples"],
            num_val_users=train_cfg["num_val_users"],
            negatives_per_user=train_cfg["negatives_per_user"],
            topk=cfg["metrics"]["topk"],
            data_source=cfg.get("data", {}).get("source", "synthetic"),
            ml1m_raw_dir=cfg.get("data", {}).get("ml1m_raw_dir"),
            rating_threshold=cfg.get("data", {}).get("rating_threshold", 0),
            sample_mode=cfg.get("train", {}).get("sample_mode", "sampled"),
            num_workers=cfg.get("loader", {}).get("num_workers", 0),
            pin_memory=cfg.get("loader", {}).get("pin_memory", False),
            persistent_workers=cfg.get("loader", {}).get("persistent_workers", False),
            prefetch_factor=cfg.get("loader", {}).get("prefetch_factor"),
            log_interval=cfg.get("train", {}).get("log_interval", 50),
            profile=cfg.get("train", {}).get("profile", False),
            sma_window=cfg.get("train", {}).get("sma_window", 50),
        ),
        device=device,
    )
    trainer.fit(num_users=model_cfg["num_users"])


if __name__ == "__main__":
    main()
