"""
dataloader.py
-------------
Hybrid pipeline: RGB from JPEG cache, flow from MP4 on-the-fly, audio from .wav.

Prerequisites:
  - Run preprocess_frames.py once to create frames_cache/{domain}/rgb/{id}/*.jpg
  - Flow MP4s at cfg.paths.flow_dir/{id}_flow_x.mp4 and _flow_y.mp4
"""

import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from torch.utils.data import Dataset, DataLoader

from config import Config
from augmentations import (
    build_waveform_augmentor, fix_audio_length,
    build_color_augmentor, get_spatial_params,
    apply_video_transforms, apply_spatial_to_flow,
)


# ============================================================
# AUDIO
# ============================================================

def load_audio(wav_path: Path, cfg: Config) -> np.ndarray:
    samples, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != cfg.audio.sample_rate:
        target_len = int(len(samples) * cfg.audio.sample_rate / sr)
        samples = np.interp(
            np.linspace(0, len(samples), target_len),
            np.arange(len(samples)), samples,
        ).astype(np.float32)
    return samples


def waveform_to_logspec(samples: np.ndarray, cfg: Config, hann_window: torch.Tensor) -> torch.Tensor:
    a   = cfg.audio
    wav = torch.from_numpy(samples).unsqueeze(0)
    spec = torch.stft(wav, n_fft=a.n_fft, hop_length=a.hop_length,
                      win_length=a.win_length, window=hann_window, return_complex=True)
    spec = torch.log(spec.abs() ** 2 + 1e-6)
    return (spec - spec.mean()) / (spec.std() + 1e-9)   # (1, F, T)


# ============================================================
# VIDEO / FLOW
# ============================================================

def _sample_indices(total: int, num: int) -> List[int]:
    if total >= num:
        return torch.linspace(0, total - 1, num).long().tolist()
    return list(range(total)) + [total - 1] * (num - total)


def load_rgb_frames(instance_dir: Path, cfg: Config) -> torch.Tensor:
    """Load from JPEG cache → (C, T, H, W) float [0,1]."""
    v = cfg.video
    files = sorted(instance_dir.glob("frame_*.jpg"))
    if not files:
        raise RuntimeError(f"No frames in {instance_dir}. Run preprocess_frames.py first.")
    frames = []
    for idx in _sample_indices(len(files), v.num_frames):
        img = cv2.imread(str(files[idx]))
        if img is None:
            raise RuntimeError(f"Failed to read {files[idx]}")
        frames.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    vid = torch.stack(frames).float() / 255.0          # (T, H, W, C)
    vid = vid.permute(3, 0, 1, 2)                      # (C, T, H, W)
    C, T, H, W = vid.shape
    scale = v.scale_size / min(H, W)
    vid_t = F.interpolate(vid.permute(1,0,2,3),
                          size=(int(H*scale), int(W*scale)),
                          mode="bilinear", align_corners=False)
    return vid_t.permute(1, 0, 2, 3)


def load_flow_from_mp4(mp4_path: Path, cfg: Config) -> torch.Tensor:
    """Decode flow MP4 → (C, T, H, W) float [-1, 1]."""
    v   = cfg.video
    vid = iio.imread(str(mp4_path), plugin="pyav")     # (T, H, W, C) uint8
    frames = []
    for idx in _sample_indices(vid.shape[0], v.num_frames):
        frames.append(torch.from_numpy((vid[idx].astype(np.float32) / 127.5) - 1.0))
    vid_t = torch.stack(frames).permute(3, 0, 1, 2)   # (C, T, H, W)
    C, T, H, W = vid_t.shape
    scale = v.scale_size / min(H, W)
    out = F.interpolate(vid_t.permute(1,0,2,3),
                        size=(int(H*scale), int(W*scale)),
                        mode="bilinear", align_corners=False)
    return out.permute(1, 0, 2, 3)


# ============================================================
# DATASET
# ============================================================

class HACDataset(Dataset):
    def __init__(self, cfg: Config, domains: List[str], split: str, verbose: bool = True):
        self.cfg      = cfg
        self.is_train = split == "train"
        self.samples  = []
        self.waveform_aug = build_waveform_augmentor(cfg) if self.is_train else None
        self.color_aug    = build_color_augmentor(cfg)    if self.is_train else None
        self._hann = torch.hann_window(cfg.audio.n_fft)

        for domain in domains:
            csv_path = (cfg.paths.train_csv if self.is_train else cfg.paths.test_csv).format(
                domain=domain, Domain=domain.capitalize()
            )
            if not Path(csv_path).exists():
                print(f"[WARN] CSV not found: {csv_path}")
                continue

            audio_dir   = Path(cfg.paths.audio_dir.format(domain=domain, Domain=domain.capitalize()))
            frames_base = Path(cfg.paths.frames_cache_dir) / domain
            flow_dir    = Path(cfg.paths.flow_dir.format(domain=domain, Domain=domain.capitalize()))

            loaded = missing = 0
            for _, row in pd.read_csv(csv_path).iterrows():
                iid   = Path(str(row.iloc[0])).stem
                label = int(row.iloc[1])
                rgb_dir     = frames_base / "rgb" / iid
                flow_x_path = flow_dir / f"{iid}_flow_x.mp4"
                flow_y_path = flow_dir / f"{iid}_flow_y.mp4"
                audio_path  = audio_dir / f"{iid}.wav"
                if not (audio_path.exists() and rgb_dir.exists()
                        and any(rgb_dir.iterdir())
                        and flow_x_path.exists() and flow_y_path.exists()):
                    missing += 1
                    continue
                self.samples.append((domain, iid, label))
                loaded += 1

            if verbose:
                print(f"[HACDataset] {domain}/{split}: {loaded} loaded, {missing} skipped")

        if not self.samples:
            raise RuntimeError(f"HACDataset empty for split={split}, domains={domains}.")
        if verbose:
            print(f"[HACDataset] Total {split}: {len(self.samples)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        domain, iid, label = self.samples[idx]
        cfg = self.cfg

        audio_dir   = Path(cfg.paths.audio_dir.format(domain=domain, Domain=domain.capitalize()))
        frames_base = Path(cfg.paths.frames_cache_dir) / domain
        flow_dir    = Path(cfg.paths.flow_dir.format(domain=domain, Domain=domain.capitalize()))

        # Audio
        samples = load_audio(audio_dir / f"{iid}.wav", cfg)
        if self.is_train and self.waveform_aug:
            samples = self.waveform_aug(samples=samples, sample_rate=cfg.audio.sample_rate)
        samples = fix_audio_length(samples, cfg.audio.sample_rate, cfg.audio.clip_duration)
        samples = samples / (np.abs(samples).max() + 1e-9)
        spec = waveform_to_logspec(samples, cfg, self._hann).repeat(3, 1, 1)  # (3, F, T)

        # RGB
        video = load_rgb_frames(frames_base / "rgb" / iid, cfg)
        _, _, H, W = video.shape
        sp    = get_spatial_params(cfg, H, W, self.is_train)
        video = apply_video_transforms(video, sp, self.color_aug,
                                       cfg.video.mean, cfg.video.std, self.is_train)

        # Flow
        flow_x = load_flow_from_mp4(flow_dir / f"{iid}_flow_x.mp4", cfg)
        flow_y = load_flow_from_mp4(flow_dir / f"{iid}_flow_y.mp4", cfg)
        flow_x, flow_y = apply_spatial_to_flow(flow_x, flow_y, sp)

        return {"audio": spec, "video": video, "flow_x": flow_x,
                "flow_y": flow_y, "label": label, "domain": domain, "id": iid}


def _collate(batch):
    return {
        "audio":  torch.stack([b["audio"]  for b in batch]),
        "video":  torch.stack([b["video"]  for b in batch]),
        "flow_x": torch.stack([b["flow_x"] for b in batch]),
        "flow_y": torch.stack([b["flow_y"] for b in batch]),
        "label":  torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "domain": [b["domain"] for b in batch],
        "id":     [b["id"]     for b in batch],
    }


def _loader(dataset, cfg, shuffle, drop_last):
    return DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and cfg.training.device == "cuda",
        collate_fn=_collate, drop_last=drop_last,
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=4 if cfg.training.num_workers > 0 else None,
    )


def build_train_loader(cfg: Config, domains: Optional[List[str]] = None) -> DataLoader:
    return _loader(HACDataset(cfg, domains or cfg.domains.train_domains, "train"), cfg, True, True)


def build_test_loader(cfg: Config, domain: str) -> DataLoader:
    return _loader(HACDataset(cfg, [domain], "test"), cfg, False, False)


def build_all_test_loaders(cfg: Config) -> dict:
    return {d: build_test_loader(cfg, d) for d in cfg.domains.test_domains}
