# utils/metrics.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F

# sklearn is optional
try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
    HAS_SK = True
except Exception:
    HAS_SK = False


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: Iterable[int] = (1,)) -> Dict[str, float]:
    """Compute top-k accuracy from logits."""
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()                                                # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = {}
    for k in ks:
        res[f"top{k}"] = correct[:k].reshape(-1).float().mean().item()
    return res


@torch.no_grad()
def confusion_matrix_torch(logits_or_preds: torch.Tensor,
                           targets: torch.Tensor,
                           num_classes: Optional[int] = None) -> torch.Tensor:
    """Confusion matrix built in pure torch. Returns CPU tensor [C, C]."""
    if logits_or_preds.dim() > 1:
        preds = logits_or_preds.argmax(dim=1)
    else:
        preds = logits_or_preds
    if num_classes is None:
        num_classes = int(torch.max(torch.stack([preds.max(), targets.max()])).item()) + 1
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=targets.device)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm.cpu()


@torch.no_grad()
def precision_recall_f1(logits: torch.Tensor,
                        targets: torch.Tensor,
                        average: str = "macro",
                        num_classes: Optional[int] = None):
    """
    Torch-native macro/micro precision/recall/F1 computed from a confusion matrix.
    Returns floats if average in {'macro','micro'}, otherwise per-class lists.
    """
    cm = confusion_matrix_torch(logits, targets, num_classes)
    tp = cm.diag().to(torch.float32)
    fp = cm.sum(dim=0).to(torch.float32) - tp
    fn = cm.sum(dim=1).to(torch.float32) - tp

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)

    if average == "macro":
        return precision.mean().item(), recall.mean().item(), f1.mean().item()
    if average == "micro":
        TP = tp.sum(); FP = fp.sum(); FN = fn.sum()
        prec = TP / (TP + FP + 1e-12)
        rec  = TP / (TP + FN + 1e-12)
        f1m  = 2 * prec * rec / (prec + rec + 1e-12)
        return prec.item(), rec.item(), f1m.item()
    # per-class
    return precision.tolist(), recall.tolist(), f1.tolist()


@torch.no_grad()
def roc_auc_from_logits(logits: torch.Tensor,
                        targets: torch.Tensor,
                        average: str = "macro",
                        multi_class: str = "ovr") -> Optional[float]:
    """
    Proper multi-class ROC-AUC using probabilities. Requires scikit-learn.
    Returns None if sklearn unavailable or computation fails.
    """
    if not HAS_SK:
        return None
    try:
        y_true = targets.detach().cpu().numpy()
        y_score = F.softmax(logits, dim=1).detach().cpu().numpy()
        return float(roc_auc_score(y_true, y_score, multi_class=multi_class, average=average))
    except Exception:
        return None


# Backward-compatible helper (now supports optional y_score for correct ROC-AUC)
def evaluate_metrics(y_true, y_pred, num_classes: int, y_score=None) -> Dict[str, Any]:
    """
    If y_score is provided (probabilities or logits), computes proper ROC-AUC.
    Otherwise, ROC-AUC is returned as None (AUC on hard labels is invalid).
    """
    out = {}
    if HAS_SK:
        out["Accuracy"] = float(accuracy_score(y_true, y_pred))
        out["F1-Score"] = float(f1_score(y_true, y_pred, average="macro"))
        if y_score is not None:
            try:
                if isinstance(y_score, torch.Tensor):
                    y_score = F.softmax(y_score, dim=1).detach().cpu().numpy()
                out["ROC-AUC"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
            except Exception:
                out["ROC-AUC"] = None
        else:
            out["ROC-AUC"] = None
    else:
        # minimal fallback without sklearn
        out["Accuracy"] = float((torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item())
        out["F1-Score"] = None
        out["ROC-AUC"] = None
    return out
