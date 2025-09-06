# main.py
from __future__ import annotations
import argparse
import sys
import subprocess

from eval import evaluate  # we can call eval directly


def add_train_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train MobileViT on your dataset")
    # data
    p.add_argument("--data-dir", type=str, default="data/PlantVillage")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="test")
    # model / input
    p.add_argument("--img-size", type=int, default=224)
    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "plateau", "none"])
    p.add_argument("--step-size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.5)
    # speed & stability
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop", type=int, default=0, help="Patience; 0 disables")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--use-class-weights", action="store_true")
    # ckpts
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save-every", type=int, default=0)
    # misc
    p.add_argument("--seed", type=int, default=42)
    return p


def add_eval_subparser(subparsers):
    p = subparsers.add_parser("eval", help="Evaluate a trained checkpoint")
    p.add_argument("--model", type=str, default="outputs/checkpoints/best.pth")
    p.add_argument("--data-dir", type=str, default="data/village_doc")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--confusion", action="store_true", help="Print confusion matrix")
    return p


def run_train_with_args(ns: argparse.Namespace):
    """
    Forward parsed args to train.py by invoking it as a subprocess.
    This avoids reworking train.py and keeps a single source of truth for flags.
    """
    cmd = [
        sys.executable, "train.py",
        "--data-dir", ns.data_dir,
        "--train-split", ns.train_split,
        "--val-split", ns.val_split,
        "--img-size", str(ns.img_size),
        "--epochs", str(ns.epochs),
        "--batch-size", str(ns.batch_size),
        "--lr", str(ns.lr),
        "--weight-decay", str(ns.weight_decay),
        "--optimizer", ns.optimizer,
        "--scheduler", ns.scheduler,
        "--step-size", str(ns.step_size),
        "--gamma", str(ns.gamma),
        "--accum-steps", str(ns.accum_steps),
        "--grad-clip", str(ns.grad_clip),
        "--workers", str(ns.workers),
        "--save-every", str(ns.save_every),
        "--seed", str(ns.seed),
    ]
    if ns.amp: cmd.append("--amp")
    if ns.pin_memory: cmd.append("--pin-memory")
    if ns.use_class_weights: cmd.append("--use-class-weights")
    if ns.early_stop and ns.early_stop > 0:
        cmd += ["--early-stop", str(ns.early_stop)]
    if ns.resume:
        cmd += ["--resume", ns.resume]

    print("[main] running:", " ".join(cmd))
    # propagate return code
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="MobileViT Plant Disease Classifier")
    subparsers = parser.add_subparsers(dest="command")

    train_p = add_train_subparser(subparsers)
    eval_p = add_eval_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "train":
        run_train_with_args(args)

    elif args.command == "eval":
        evaluate(
            model_path=args.model,
            data_dir=args.data_dir,
            split=args.split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            show_confusion=args.confusion,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
