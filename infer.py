# infer.py
from __future__ import annotations
import argparse, time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.mobilevit import MobileViTClassifier


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def inference_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def find_classes_from_dir(data_root: Path) -> list[str]:
    """
    Infer class names from dataset folder layout.
    Priority: train/ → val/ → test/ → flat dir.
    """
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if split_dir.is_dir():
            return sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    # fallback: flat class folders directly under data_root
    return sorted([d.name for d in data_root.iterdir() if d.is_dir()])

def load_model(model_path: str, num_classes: int, img_size: int, device: torch.device):
    model = MobileViTClassifier(image_size=(img_size, img_size), num_classes=num_classes).to(device)
    ckpt = torch.load(model_path, map_location=device)

    # robust load: support our trainer's dict or a raw state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # might already be the raw sd
        state_dict = ckpt
    else:
        raise ValueError("Unrecognized checkpoint format.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

@torch.no_grad()
def predict_one(model, image_path: Path, classes: list[str], img_size: int, device: torch.device, topk: int = 5):
    tfm = inference_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    dt = (time.time() - t0) * 1000

    k = min(topk, len(classes))
    conf, idx = probs.topk(k)
    idx = idx.tolist(); conf = conf.tolist()
    return dt, [(classes[i], conf[j]) for j, i in enumerate(idx)]

def parse_args():
    p = argparse.ArgumentParser(description="Single-image inference with MobileViT")
    p.add_argument("--model", type=str, required=True, help="Path to checkpoint (e.g., outputs/checkpoints/best.pth)")
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--data-dir", type=str, required=True, help="Dataset root to infer class names (e.g., data/plantvillage)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    data_root = Path(args.data_dir)
    classes = find_classes_from_dir(data_root)
    if not classes:
        raise RuntimeError(f"No classes found under {data_root}. Make sure folders exist under train/ (or val/test).")

    model = load_model(args.model, num_classes=len(classes), img_size=args.img_size, device=device)
    dt_ms, preds = predict_one(model, Path(args.image), classes, img_size=args.img_size, device=device, topk=args.topk)

    print(f"Inference time: {dt_ms:.1f} ms  (img_size={args.img_size}, device={device})")
    for rank, (name, p) in enumerate(preds, 1):
        print(f"Top-{rank}: {name:55s}  {p*100:6.2f}%")

if __name__ == "__main__":
    main()
