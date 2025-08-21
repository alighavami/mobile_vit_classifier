# utils/transformrs.py
from __future__ import annotations
from torchvision import transforms


# ImageNet statistics (works well for MobileViT pretrained weights)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(
    img_size: int = 224,
    aug_level: str = "medium",   # "light" | "medium" | "strong"
    normalize_mean=IMAGENET_MEAN,
    normalize_std=IMAGENET_STD,
):
    """
    Returns (train_tfms, val_tfms) torchvision transform pipelines.
    - 'light'  : minimal augmentations, quick convergence
    - 'medium' : recommended default for field images
    - 'strong' : heavier spatial/color aug for stronger regularization
    """
    # Common
    norm = transforms.Normalize(normalize_mean, normalize_std)

    if aug_level not in {"light", "medium", "strong"}:
        aug_level = "medium"

    if aug_level == "light":
        train_tfms = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(int(img_size * 1.05)),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            norm,
        ])
    elif aug_level == "medium":
        train_tfms = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), inplace=True),
        ])
    else:  # strong
        train_tfms = transforms.Compose([
            transforms.Resize(int(img_size * 1.20)),
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.08)], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3), inplace=True),
        ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])

    return train_tfms, val_tfms


def inference_transform(img_size: int = 224, normalize_mean=IMAGENET_MEAN, normalize_std=IMAGENET_STD):
    """
    Deterministic transform for evaluation/inference (single-crop).
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])


def denormalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Undo normalization for visualization.
    t: Tensor in shape [C, H, W] normalized by mean/std.
    """
    import torch
    if not isinstance(t, torch.Tensor):
        return t
    mean = t.new_tensor(mean)[:, None, None]
    std = t.new_tensor(std)[:, None, None]
    return t * std + mean
