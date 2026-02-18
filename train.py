from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml

from onerec.data.ml1m import infer_ml1m_num_items
from onerec.models.openonerec_like import OpenOneRecLikeConfig, OpenOneRecLikeModel
from onerec.training.trainer import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenOneRec-like PyTorch training entrypoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    dataset_type = train_cfg.get("dataset_type", "synthetic")

    num_items = model_cfg.get("num_items", 0)
    if dataset_type == "ml1m":
        ratings_path = train_cfg.get("ratings_path")
        if not ratings_path:
            raise ValueError("train.ratings_path is required when dataset_type=ml1m")
        num_items = infer_ml1m_num_items(ratings_path)

    model = OpenOneRecLikeModel(
        OpenOneRecLikeConfig(
            num_items=num_items,
            max_seq_len=model_cfg["max_seq_len"],
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            n_heads=model_cfg["n_heads"],
            num_decoder_layers=model_cfg["num_decoder_layers"],
            ffn_dim=model_cfg["ffn_dim"],
            num_experts=model_cfg["num_experts"],
            dropout=model_cfg["dropout"],
        )
    )

    trainer = Trainer(
        model=model,
        cfg=TrainConfig(
            seed=cfg["seed"],
            dataset_type=dataset_type,
            objective=train_cfg.get("objective", "generative_ce"),
            num_items=num_items,
            max_seq_len=model_cfg["max_seq_len"],
            batch_size=train_cfg["batch_size"],
            epochs=train_cfg["epochs"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            num_train_samples=train_cfg.get("num_train_samples", 0),
            num_val_users=train_cfg.get("num_val_users", 0),
            negatives_per_user=train_cfg["negatives_per_user"],
            topk=cfg["metrics"]["topk"],
            ratings_path=train_cfg.get("ratings_path"),
            eval_mode=train_cfg.get("eval_mode", "sampled"),
            filter_seen_in_eval=train_cfg.get("filter_seen_in_eval", True),
            dpo_beta=train_cfg.get("dpo_beta", 0.1),
            bos_token=train_cfg.get("bos_token", 1),
        ),
        device=device,
    )
    trainer.fit(num_users=model_cfg.get("num_users", 0))


if __name__ == "__main__":
    main()
