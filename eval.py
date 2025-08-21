# eval.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# optional sklearn metrics
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SK = True
except Exception:
    HAS_SK = False

from datasets.plant_dataset import PlantDataset
from models.mobilevit import MobileViTClassifier
from utils.transformrs import inference_transform


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MobileViT plant disease classifier")
    p.add_argument("--model", type=str, default="outputs/checkpoints/best.pth",
                   help="Path to checkpoint or raw state_dict .pth")
    p.add_argument("--data-dir", type=str, default="data/PlantVillage",
                   help="Root dataset directory (with splits inside)")
    p.add_argument("--split", type=str, default="test",
                   help="Which split subfolder to evaluate (e.g., test/val)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--confusion", action="store_true",
                   help="Print confusion matrix (requires sklearn)")
    return p.parse_args()


@torch.no_grad()
def evaluate(model_path: str, data_dir: str, split: str, img_size: int,
             batch_size: int, workers: int, device: str, show_confusion: bool):
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Dataset & loader (infer classes from folders)
    tfm = inference_transform(img_size=img_size)
    ds = PlantDataset(root=data_dir, split=split, class_map=None, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(dev.type == "cuda"))

    num_classes = len(ds.classes)

    # Build model
    model = MobileViTClassifier(image_size=(img_size, img_size), num_classes=num_classes).to(dev)
    model.eval()

    # Load weights (support both raw state_dict and our trainer checkpoint)
    ckpt = torch.load(model_path, map_location=dev)
    state_dict = None
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and all(k.startswith(("model.", "conv1", "mv2", "mvit", "fc", "conv2"))
                                        for k in ckpt.keys()):
        state_dict = ckpt  # looks like a raw state dict
    else:
        # try common nesting keys
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))

    model.load_state_dict(state_dict, strict=True)

    # Eval loop
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(dev, non_blocking=True)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

    # Metrics
    acc = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()
    print(f"Accuracy: {acc:.4f}  (N={len(ds)})")

    if HAS_SK:
        print(classification_report(y_true, y_pred, target_names=ds.classes, digits=4))
        if show_confusion:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            print("Confusion matrix (rows=true, cols=pred):")
            print(cm)
    else:
        print("(Install scikit-learn for per-class metrics and confusion matrix.)")

    return acc


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.model).exists():
        print(f"[Error] Model file not found: {args.model}")
        sys.exit(1)
    evaluate(
        model_path=args.model,
        data_dir=args.data_dir,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        show_confusion=args.confusion
    )
