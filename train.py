"""
train.py
--------
Training loop — architecture-agnostic.
All architecture decisions live in pipeline.py.

Usage:
    python train.py                                 # default config
    python train.py --pipeline clip_multimodal      # CLIP contrastive pipeline
    python train.py --pipeline concat_multimodal    # switch architecture
    python train.py --epochs 200 --lr 5e-4
    python train.py --resume outputs/checkpoint.pt --epochs 200
    python train.py --set training.batch_size=32 domains.train_domains='["human"]'

Note on clip_multimodal:
    The loss function is InfoNCE (contrastive cross-entropy).
    build_loss() returns an InfoNCELoss instance whose signature is
    loss_fn(logits, labels) — identical to CrossEntropyLoss — so
    the training loop below requires NO special-casing.
"""

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
# import torch

from config import get_config
from pipeline import build_pipeline, build_optimizer, build_scheduler, build_loss
from dataloader import build_train_loader, build_all_test_loaders
from metrics import compute_metrics
import torch
torch.backends.cudnn.enabled = False

# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_ckpt(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] → {path}")


# ============================================================
# EPOCH LOOPS
# ============================================================

def train_epoch(model, loader, optimizer, loss_fn, device, cfg):
    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []
    for i, batch in enumerate(loader):
        audio  = batch["audio"].to(device,  non_blocking=True)
        video  = batch["video"].to(device,  non_blocking=True)
        flow_x = batch["flow_x"].to(device, non_blocking=True)
        flow_y = batch["flow_y"].to(device, non_blocking=True)
        labels = batch["label"].to(device,  non_blocking=True)

        optimizer.zero_grad()
        logits = model(audio, video, flow_x, flow_y)
        # loss_fn(logits, labels) works for both CrossEntropy and InfoNCELoss
        loss   = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            # Log temperature if available (clip_multimodal)
            temp_str = ""
            if hasattr(model, "log_tau"):
                temp_str = f"  τ={model.log_tau.exp().item():.4f}"
            print(f"    batch [{i+1}/{len(loader)}] loss={loss.item():.4f}{temp_str}", end="\r")
    print()

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels),
                              cfg.training.top_k)
    return total_loss / len(loader.dataset), metrics


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, cfg):
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for batch in loader:
        audio  = batch["audio"].to(device,  non_blocking=True)
        video  = batch["video"].to(device,  non_blocking=True)
        flow_x = batch["flow_x"].to(device, non_blocking=True)
        flow_y = batch["flow_y"].to(device, non_blocking=True)
        labels = batch["label"].to(device,  non_blocking=True)
        logits = model(audio, video, flow_x, flow_y)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        all_logits.append(logits.cpu()); all_labels.append(labels.cpu())
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels),
                              cfg.training.top_k)
    return total_loss / len(loader.dataset), metrics


# ============================================================
# MAIN
# ============================================================

def train(cfg, resume=None):
    set_seed(cfg.training.seed)
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU"); device = "cpu"

    t0_run   = time.perf_counter()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir  = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(cfg.domains.train_domains)

    print(f"\n{'='*60}")
    print(f"  Pipeline:  {cfg.pipeline}")
    print(f"  Domains:   {cfg.domains.train_domains} → {cfg.domains.test_domains}")
    print(f"  Device:    {device}  |  Started: {run_time}")
    if cfg.pipeline == "clip_multimodal":
        print(f"  Text enc:  {cfg.clip.model_name}")
        print(f"  Classes:   {cfg.clip.action_classes}")
        print(f"  Loss:      InfoNCE (τ_init={cfg.clip.init_temp}, learnable={cfg.clip.learn_temp})")
    print(f"{'='*60}\n")

    train_loader = build_train_loader(cfg)
    test_loaders = build_all_test_loaders(cfg)

    model     = build_pipeline(cfg).to(device)
    loss_fn   = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, cfg.training.epochs)

    best_acc, start_epoch, log = 0.0, 0, []

    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_acc    = ckpt.get("best_acc", 0.0)
        if scheduler:
            for _ in range(start_epoch): scheduler.step()
        print(f"[Resume] epoch {start_epoch}, best_acc={best_acc:.4f}")

    for epoch in range(start_epoch, cfg.training.epochs):
        if device == "cuda": torch.cuda.empty_cache()
        t0 = time.perf_counter()

        tr_loss, tr_m = train_epoch(model, train_loader, optimizer, loss_fn, device, cfg)

        if scheduler:
            scheduler.step(tr_m["top1"]) if cfg.training.scheduler == "plateau" else scheduler.step()

        lr = optimizer.param_groups[-1]["lr"]
        print(f"Epoch [{epoch+1:03d}/{cfg.training.epochs}]  "
              f"loss={tr_loss:.4f}  top1={tr_m['top1']:.4f}  "
              f"mca={tr_m['mca']:.4f}  lr={lr:.2e}  "
              f"t={time.perf_counter()-t0:.1f}s")

        if (epoch + 1) % cfg.training.eval_every == 0 or epoch == cfg.training.epochs - 1:
            row = {"epoch": epoch+1, "train_loss": tr_loss, "train_top1": tr_m["top1"]}
            for d, loader in test_loaders.items():
                _, m = eval_epoch(model, loader, loss_fn, device, cfg)
                print(f"    {d:8s} → top1={m['top1']:.4f}  mca={m['mca']:.4f}")
                row[f"{d}_top1"] = m["top1"]; row[f"{d}_mca"] = m["mca"]

            avg = sum(row[f"{d}_top1"] for d in test_loaders) / len(test_loaders)
            row["avg_top1"] = avg; log.append(row)

            if avg > best_acc and cfg.training.save_best:
                best_acc = avg
                save_ckpt({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "epoch": epoch+1, "best_acc": best_acc, "cfg": cfg},
                           out_dir / f"best_{cfg.pipeline}_{tag}.pt")

        if (epoch + 1) % cfg.training.save_every == 0:
            save_ckpt({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch+1, "best_acc": best_acc, "cfg": cfg},
                       out_dir / f"ckpt_epoch{epoch+1:03d}_{tag}.pt")

    pd.DataFrame(log).to_csv(out_dir / "train_results.csv", index=False)
    total = time.perf_counter() - t0_run
    print(f"\n[Done] {total/60:.1f} min | best avg top1={best_acc:.4f}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline",   type=str, default=None, help="pipeline name from pipeline.py")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--resume",     type=str,   default=None)
    parser.add_argument("--set",        nargs="*",  default=[],
                        help="Override any config field: --set training.lr=1e-4 pipeline=clip_multimodal")
    args = parser.parse_args()

    cfg = get_config(pipeline=args.pipeline or "clip_multimodal")
    if args.epochs:     cfg.training.epochs     = args.epochs
    if args.lr:         cfg.training.lr         = args.lr
    if args.batch_size: cfg.training.batch_size = args.batch_size

    for kv in (args.set or []):
        k, v = kv.split("=", 1)
        import ast
        try:    v = ast.literal_eval(v)
        except: pass
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]: obj = getattr(obj, p)
        setattr(obj, parts[-1], v)

    train(cfg, resume=args.resume)
