# config.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import argparse


@dataclass
class Config:
    # -------------------------
    # Data
    # -------------------------
    data_dir: str = "data"
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: Optional[str] = None  # optional
    num_classes: Optional[int] = None  # will be inferred from folders if None
    img_size: int = 224

    # -------------------------
    # Model
    # -------------------------
    model_variant: str = "mobilevit_xxs"  # mobilevit_xxs | mobilevit_xs | mobilevit_s
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    label_smoothing: float = 0.0

    # -------------------------
    # Training
    # -------------------------
    epochs: int = 50
    batch_size: int = 64
    optimizer: str = "adamw"            # adam | sgd | adamw
    lr: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9               # for SGD
    betas: tuple = (0.9, 0.999)         # for Adam/AdamW

    # Scheduler
    scheduler: str = "cosine"           # cosine | step | none
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    step_size: int = 30                 # for StepLR
    gamma: float = 0.1                  # for StepLR

    # Mixed precision / stability
    amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    grad_accum_steps: int = 1

    # Early stopping / EMA (optional hooks in trainer)
    early_stop_patience: Optional[int] = None
    use_ema: bool = False
    ema_decay: float = 0.9999

    # -------------------------
    # System
    # -------------------------
    device: str = "cuda"                # "cuda" or "cpu" (trainer will fallback to CPU if CUDA not available)
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    cudnn_benchmark: bool = True
    deterministic: bool = False

    # -------------------------
    # Logging / Checkpoints
    # -------------------------
    out_dir: str = "outputs"
    ckpt_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    save_every: int = 0                 # save every N epochs; 0 => only best
    save_best_only: bool = True
    resume: Optional[str] = None        # path to checkpoint to resume from

    # -------------------------
    # DDP (future-proof; safe to ignore if single-GPU)
    # -------------------------
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    dist_backend: str = "nccl"
    dist_url: str = "env://"            # read MASTER_ADDR/MASTER_PORT, etc.

    # Free-form notes (useful for experiment tracking)
    notes: str = ""

    # Internal: store class names after dataloader build (don’t set manually)
    class_names: list[str] = field(default_factory=list)

    # ------------ utility methods ------------
    def make_dirs(self) -> None:
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save_json(self, path: str | os.PathLike) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load_json(path: str | os.PathLike) -> "Config":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        # convert tuples if they got saved as lists
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])
        return Config(**data)

    def update_from_env(self) -> None:
        """Optional: allow quick overrides from environment variables."""
        self.data_dir = os.getenv("DATA_DIR", self.data_dir)
        self.device = os.getenv("DEVICE", self.device)
        self.out_dir = os.getenv("OUT_DIR", self.out_dir)
        self.ckpt_dir = os.getenv("CKPT_DIR", self.ckpt_dir)
        self.log_dir = os.getenv("LOG_DIR", self.log_dir)

    @staticmethod
    def from_args() -> "Config":
        """Create a Config from CLI args; all args are optional overrides."""
        parser = argparse.ArgumentParser(description="MobileViT training config")
        add = parser.add_argument

        # Only the most common flags exposed; advanced ones can be edited in file.
        add("--data_dir", type=str)
        add("--train_dir", type=str)
        add("--val_dir", type=str)
        add("--num_classes", type=int)
        add("--img_size", type=int)

        add("--model_variant", type=str, choices=["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"])
        add("--pretrained", type=lambda x: x.lower() == "true")

        add("--epochs", type=int)
        add("--batch_size", type=int)
        add("--optimizer", type=str, choices=["adam", "sgd", "adamw"])
        add("--lr", type=float)
        add("--weight_decay", type=float)
        add("--momentum", type=float)
        add("--betas", type=float, nargs=2)

        add("--scheduler", type=str, choices=["cosine", "step", "none"])
        add("--warmup_epochs", type=int)
        add("--min_lr", type=float)
        add("--step_size", type=int)
        add("--gamma", type=float)

        add("--amp", type=lambda x: x.lower() == "true")
        add("--grad_clip_norm", type=float)
        add("--grad_accum_steps", type=int)
        add("--label_smoothing", type=float)

        add("--device", type=str, choices=["cuda", "cpu"])
        add("--num_workers", type=int)
        add("--pin_memory", type=lambda x: x.lower() == "true")
        add("--seed", type=int)

        add("--out_dir", type=str)
        add("--ckpt_dir", type=str)
        add("--log_dir", type=str)
        add("--save_every", type=int)
        add("--save_best_only", type=lambda x: x.lower() == "true")
        add("--resume", type=str)

        add("--notes", type=str)

        args = parser.parse_args()
        cfg = Config()
        # apply CLI overrides if provided
        for k, v in vars(args).items():
            if v is not None:
                setattr(cfg, k, v)

        cfg.update_from_env()
        cfg.make_dirs()
        return cfg


if __name__ == "__main__":
    # Allows: python config.py --epochs 10 --batch_size 32 ...
    cfg = Config.from_args()
    print(json.dumps(cfg.to_dict(), indent=2))
