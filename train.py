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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    model = SimpleSequentialRec(
        num_items=model_cfg["num_items"],
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
        ),
        device=device,
    )
    trainer.fit(num_users=model_cfg["num_users"])


if __name__ == "__main__":
    main()
