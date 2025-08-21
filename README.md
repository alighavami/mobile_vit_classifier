# MobileViT Plant Disease Classifier

Lightweight MobileViT for plant disease classification, with CUDA-ready Docker training, AMP, grad-accum, early stopping, clean metrics, and reproducible results.

---

## ✅ Features
- Pure PyTorch implementation of **MobileViT** (no CBAM, per your request)
- **CUDA/AMP** training support (mixed precision)
- **Docker** GPU workflow (CUDA 11.8 runtime)
- Robust trainer: grad clipping, accumulation, resume, best/last checkpoints, CSV logs
- Clean dataset loader (ImageFolder-style), class weights for imbalance
- Simple CLI via `main.py` (`train` / `eval`)

---

## 📂 Project Structure
