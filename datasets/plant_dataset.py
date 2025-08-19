# plant_dataset.py
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class PlantDataset(Dataset):
    def __init__(self, root, split, class_map, transform=None):
        self.root = os.path.join(root, split)
        assert os.path.isdir(self.root), f"No directory at {self.root}"
        
        self.class_map = class_map
        self.transform = transform
        
        # Ensure class order is deterministic
        self.classes = sorted([d for d in os.listdir(self.root)
                               if os.path.isdir(os.path.join(self.root, d))])
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for cls in self.classes:
            cls_idx = self.class_map[cls]
            cls_path = os.path.join(self.root, cls)
            for img in os.listdir(cls_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples.append((os.path.join(cls_path, img), cls_idx))
        assert samples, f"No images found in {self.root}"
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
