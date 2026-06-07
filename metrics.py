"""
metrics.py — Top-1, Top-K, Mean Class Accuracy.
"""
import torch
from typing import Dict


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor,
                    top_k: int = 5) -> Dict[str, float]:
    preds   = logits.argmax(dim=1)
    top1    = (preds == labels).float().mean().item()
    k       = min(top_k, logits.size(1))
    topk    = (logits.topk(k, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    num_cls = logits.size(1)
    cc      = torch.zeros(num_cls); ct = torch.zeros(num_cls)
    for c in range(num_cls):
        mask = labels == c; ct[c] = mask.sum(); cc[c] = (preds[mask] == c).sum()
    mca = (cc / (ct + 1e-9)).mean().item()
    return {"top1": top1, f"top{k}": topk, "mca": mca}
