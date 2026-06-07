"""
test.py
-------
Evaluate a checkpoint on all (or one) test domain(s).

Usage:
    python test.py --checkpoint outputs/best_base_multimodal_human.pt
    python test.py --checkpoint outputs/best_clip_multimodal_human.pt --domain animal

For clip_multimodal:
    Inference is zero-shot style: the model computes cosine similarity between the
    fused visual embedding and all N class text embeddings, then argmax predicts the class.
    No special-casing needed here — model.forward() already returns (B, N) logits.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from config import get_config
from pipeline import build_pipeline, build_loss
from dataloader import build_test_loader, build_all_test_loaders
from metrics import compute_metrics


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg):
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for batch in loader:
        audio  = batch["audio"].to(device,  non_blocking=True)
        video  = batch["video"].to(device,  non_blocking=True)
        flow_x = batch["flow_x"].to(device, non_blocking=True)
        flow_y = batch["flow_y"].to(device, non_blocking=True)
        labels = batch["label"].to(device,  non_blocking=True)
        # For clip_multimodal: logits = cosine_sim(z_vis, z_txt) / tau  →  (B, N)
        # For all other pipelines: logits = classifier output            →  (B, N)
        # In both cases argmax gives the predicted class — no change needed.
        logits = model(audio, video, flow_x, flow_y)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        all_logits.append(logits.cpu()); all_labels.append(labels.cpu())
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels), cfg.training.top_k)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def test(checkpoint_path: str, domain: str = None):
    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg    = ckpt.get("cfg", get_config())
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available(): device = "cpu"

    print(f"\n[Test] pipeline={cfg.pipeline}  ckpt_epoch={ckpt.get('epoch','?')}  "
          f"best_acc={ckpt.get('best_acc', 0.0):.4f}")

    if cfg.pipeline == "clip_multimodal":
        print(f"[Test] text_encoder={cfg.clip.model_name}")
        print(f"[Test] classes={cfg.clip.action_classes}")
        print(f"[Test] Inference: argmax of cosine similarity to class text embeddings")

    model = build_pipeline(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    loss_fn = build_loss(cfg)

    loaders = {domain: build_test_loader(cfg, domain)} if domain else build_all_test_loaders(cfg)

    rows = []
    t0_run = time.perf_counter()
    for d, loader in loaders.items():
        t0 = time.perf_counter()
        m  = evaluate(model, loader, loss_fn, device, cfg)
        print(f"  {d:10s} top1={m['top1']:.4f}  mca={m['mca']:.4f}  "
              f"loss={m['loss']:.4f}  t={time.perf_counter()-t0:.1f}s")
        rows.append({"domain": d, **m})

    out = Path(checkpoint_path).parent / "test_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[Test] Done in {time.perf_counter()-t0_run:.1f}s  → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain",     default=None)
    args = parser.parse_args()
    test(args.checkpoint, args.domain)
