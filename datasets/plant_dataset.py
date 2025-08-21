# datasets/plant_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps


def _is_image_file(p: Path, exts: Sequence[str]) -> bool:
    return p.suffix.lower() in exts and not p.name.startswith(".")


class PlantDataset(Dataset):
    """
    Directory structure expected (ImageFolder-style):

        root/
          <split>/
            class_a/
              img1.jpg ...
            class_b/
              ...

    Args:
        root: Root directory containing the split subdirectory.
        split: Subfolder under root, e.g. 'train' or 'val'. If None/'' -> use root directly.
        class_map: Optional mapping {class_name: index}. If None, inferred from subfolders (sorted).
        transform: Optional torchvision transform callable.
        extensions: Allowed file extensions (lowercase).
        recursive: If True, search class folders recursively for images.
        return_paths: If True, __getitem__ returns (image, label, path_str).
        strict: If True, raise on missing/unknown classes; else skip gracefully.

    Returns:
        (image_tensor, label_tensor) or (image_tensor, label_tensor, path_str)
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Optional[str],
        class_map: Optional[Dict[str, int]],
        transform=None,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        recursive: bool = False,
        return_paths: bool = False,
        strict: bool = True,
    ):
        base = Path(root)
        self.root = (base / split) if split else base
        if not self.root.is_dir():
            raise FileNotFoundError(f"No directory at {self.root}")

        self.transform = transform
        self.extensions = tuple(e.lower() for e in extensions)
        self.recursive = recursive
        self.return_paths = return_paths
        self.strict = strict

        # Discover classes from subdirectories
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")])
        if not classes:
            raise RuntimeError(f"No class subdirectories found under {self.root}")

        # Build / validate class_map
        if class_map is None:
            self.class_map = {c: i for i, c in enumerate(classes)}
        else:
            # Validate provided mapping
            missing = [c for c in classes if c not in class_map]
            if missing and strict:
                raise ValueError(f"class_map missing classes found on disk: {missing}")
            # Filter to only known classes (sorted for determinism)
            self.class_map = {c: class_map[c] for c in classes if c in class_map}

        # Keep a deterministic class list ordered by index, then name
        self.classes = [c for c, _ in sorted(self.class_map.items(), key=lambda kv: (kv[1], kv[0]))]

        # Build sample list
        self.samples: List[Tuple[Path, int]] = self._load_samples()
        if not self.samples:
            raise RuntimeError(f"No images found in {self.root} with extensions {self.extensions}")

    def _iter_class_images(self, cls_dir: Path) -> Iterable[Path]:
        if self.recursive:
            yield from (p for p in cls_dir.rglob("*") if p.is_file() and _is_image_file(p, self.extensions))
        else:
            yield from (p for p in cls_dir.iterdir() if p.is_file() and _is_image_file(p, self.extensions))

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_idx = self.class_map[cls]
            cls_path = self.root / cls
            if not cls_path.is_dir():
                if self.strict:
                    raise FileNotFoundError(f"Class folder missing: {cls_path}")
                else:
                    continue
            imgs = sorted(self._iter_class_images(cls_path), key=lambda p: p.as_posix())
            samples.extend((p, cls_idx) for p in imgs)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        # Robust load with EXIF orientation fix
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        if self.return_paths:
            return img, label, str(path)
        return img, label

    # ----------- optional helpers -----------
    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights (useful for imbalanced datasets).
        """
        import numpy as np

        counts = np.zeros(len(self.classes), dtype=np.int64)
        for _, y in self.samples:
            counts[y] += 1
        counts = np.maximum(counts, 1)
        inv = 1.0 / counts
        weights = inv / inv.sum() * len(self.classes)
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        lines = [
            f"{self.__class__.__name__}(",
            f"  root={self.root}",
            f"  classes={len(self.classes)}",
            f"  samples={len(self.samples)}",
            f"  recursive={self.recursive}, exts={self.extensions}",
            f")",
        ]
        return "\n".join(lines)
