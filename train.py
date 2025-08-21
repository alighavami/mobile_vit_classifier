# train.py
import os
import argparse
import random
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.mobilevit import MobileViTClassifier
from datasets.plant_dataset import PlantDataset
from utils.transformrs import build_transforms
from utils.trainer import trainer as Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train MobileViT for Plant Disease Classification")

    # data
    p.add_argument('--data-dir', type=str, default='data/PlantVillage')
    p.add_argument('--train-split', type=str, default='train')
    p.add_argument('--val-split', type=str, default='test')

    # model / input
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--variant', type=str, default='small', choices=['small'])  # placeholder if you add more
    p.add_argument('--pretrained', action='store_true')  # reserved flag if you add pretrained later

    # training
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'])
    p.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'none'])
    p.add_argument('--step-size', type=int, default=10)
    p.add_argument('--gamma', type=float, default=0.5)

    # stability & speed
    p.add_argument('--amp', action='store_true', help='enable mixed precision')
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--early-stop', type=int, default=0, help='patience; 0 disables')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--pin-memory', action='store_true')
    p.add_argument('--use-class-weights', action='store_true')

    # ckpts / resume
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save-every', type=int, default=0, help='save epoch checkpoints; 0 => only best/last')

    # misc
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_optimizer(params, name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    name = name.lower()
    if name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, name: str, epochs: int, step_size: int, gamma: float):
    name = name.lower()
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == 'step':
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=3)
    if name == 'none':
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = bool(args.pin_memory) and torch.cuda.is_available()

    # Transforms
    train_tf, val_tf = build_transforms(img_size=args.img_size, aug_level="medium")

    # Datasets (class_map=None => infer classes from folders, sorted)
    train_ds = PlantDataset(args.data_dir, args.train_split, class_map=None, transform=train_tf)
    val_ds   = PlantDataset(args.data_dir, args.val_split,   class_map=None, transform=val_tf)
    num_classes = len(train_ds.classes)

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_mem
    )

    # Model
    model = MobileViTClassifier(image_size=(args.img_size, args.img_size),
                                num_classes=num_classes).to(device)

    # Criterion (optional class weights for imbalance)
    if args.use_class_weights and hasattr(train_ds, "class_weights"):
        weights = train_ds.class_weights().to(device)
    else:
        weights = None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # Optimizer / Scheduler
    optimizer = build_optimizer(model.parameters(), args.optimizer, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.step_size, args.gamma)

    # Trainer
    tr = Trainer(
        model, optimizer, criterion, scheduler,
        train_loader, val_loader, device,
        output_dir='outputs',
        resume_ckpt=args.resume,
        amp=args.amp,
        grad_clip_norm=(args.grad_clip if args.grad_clip and args.grad_clip > 0 else None),
        accum_steps=max(1, args.accum_steps),
        early_stopping_patience=(args.early_stop if args.early_stop > 0 else None),
        save_every=args.save_every
    )

    tr.train(args.epochs)


if __name__ == "__main__":
    main()
