"""
utils/augmentations.py
----------------------
All augmentation logic in one place.
- Audio: waveform-level (pre-spectrogram) via audiomentations (CPU/numpy)
- Video: photometric (RGB only) + geometric (shared RGB+flow params)
- Flow:  geometric only, with flow-vector corrections
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Tuple, Optional


# ============================================================
# AUDIO AUGMENTATION (waveform, numpy)
# ============================================================

def build_waveform_augmentor(cfg):
    """
    Builds an audiomentations Compose pipeline from AudioConfig.
    Returns None if cfg.audio.augment is False.
    Call: augmentor(samples=np_array, sample_rate=sr) → np_array
    """
    if not cfg.audio.augment:
        return None

    try:
        from audiomentations import (
            Compose, AddGaussianNoise, TimeStretch,
            PitchShift, Shift, Gain
        )
    except ImportError:
        raise ImportError(
            "audiomentations not installed. Run: pip install audiomentations"
        )

    a = cfg.audio
    return Compose([
        AddGaussianNoise(
            min_amplitude=a.aug_gaussian_min,
            max_amplitude=a.aug_gaussian_max,
            p=a.aug_gaussian_noise_p,
        ),
        TimeStretch(
            min_rate=a.aug_time_stretch_min,
            max_rate=a.aug_time_stretch_max,
            p=a.aug_time_stretch_p,
        ),
        PitchShift(
            min_semitones=a.aug_pitch_min_semitones,
            max_semitones=a.aug_pitch_max_semitones,
            p=a.aug_pitch_shift_p,
        ),
        Shift(
            min_shift=a.aug_shift_min,
            max_shift=a.aug_shift_max,
            p=a.aug_shift_p,
        ),
        Gain(
            min_gain_db=a.aug_gain_min_db,
            max_gain_db=a.aug_gain_max_db,
            p=a.aug_gain_p,
        ),
    ])


def fix_audio_length(samples: np.ndarray, sample_rate: int, clip_duration: float) -> np.ndarray:
    """
    Truncate or pad waveform to a fixed length.
    Required after TimeStretch which changes duration.
    """
    target = int(sample_rate * clip_duration)
    if len(samples) >= target:
        return samples[:target]
    return np.pad(samples, (0, target - len(samples)), mode="constant")


# ============================================================
# SPATIAL AUGMENTATION PARAMS (shared RGB + flow)
# ============================================================

class SpatialAugParams:
    """
    Draws random spatial augmentation parameters once per sample.
    These same params are applied to both RGB frames and flow maps
    to preserve spatial correspondence.
    """
    def __init__(
        self,
        do_random_crop: bool,
        crop_size: int,
        frame_h: int,
        frame_w: int,
        hflip_p: float,
    ):
        # Random crop window
        if do_random_crop:
            top  = random.randint(0, frame_h - crop_size)
            left = random.randint(0, frame_w - crop_size)
        else:
            # Centre crop
            top  = (frame_h - crop_size) // 2
            left = (frame_w - crop_size) // 2

        self.top       = top
        self.left      = left
        self.crop_size = crop_size
        self.do_hflip  = random.random() < hflip_p


def get_spatial_params(cfg, frame_h: int, frame_w: int, is_train: bool) -> SpatialAugParams:
    """
    Returns SpatialAugParams for a given frame size.
    Training: random crop + random flip.
    Validation/test: centre crop + no flip.
    """
    return SpatialAugParams(
        do_random_crop=is_train and cfg.video.augment_spatial,
        crop_size=cfg.video.crop_size,
        frame_h=frame_h,
        frame_w=frame_w,
        hflip_p=cfg.video.random_hflip_p if (is_train and cfg.video.augment_spatial) else 0.0,
    )


# ============================================================
# VIDEO (RGB) AUGMENTATION
# ============================================================

def build_color_augmentor(cfg):
    """
    Builds a torchvision ColorJitter + RandomGrayscale transform.
    Applied to RGB frames ONLY, never to flow.
    Returns None if augment_color is False.
    """
    if not cfg.video.augment_color:
        return None

    import torchvision.transforms as T
    v = cfg.video
    return T.Compose([
        T.RandomApply([
            T.ColorJitter(
                brightness=v.color_jitter_brightness,
                contrast=v.color_jitter_contrast,
                saturation=v.color_jitter_saturation,
                hue=v.color_jitter_hue,
            )
        ], p=v.color_jitter_p),
        T.RandomGrayscale(p=v.grayscale_p),
    ])


def apply_spatial_to_rgb(
    frame: torch.Tensor,          # (C, H, W) float tensor
    params: SpatialAugParams,
) -> torch.Tensor:
    """Applies crop + optional hflip to a single RGB frame tensor."""
    frame = TF.crop(frame, params.top, params.left, params.crop_size, params.crop_size)
    if params.do_hflip:
        frame = TF.hflip(frame)
    return frame


def apply_video_transforms(
    video: torch.Tensor,          # (C, T, H, W) float [0,1]
    params: SpatialAugParams,
    color_augmentor,
    mean: list,
    std: list,
    is_train: bool,
) -> torch.Tensor:
    """
    Full RGB video transform pipeline:
      1. Spatial (crop + flip) — shared params
      2. Photometric (color jitter + grayscale) — RGB only, train only
      3. Normalize with Kinetics mean/std
    Input:  (C, T, H, W) float [0, 1]
    Output: (C, T, H, W) float normalized
    """
    C, T, H, W = video.shape

    # -- Spatial: apply per-frame --
    frames = []
    for t in range(T):
        f = video[:, t, :, :]                            # (C, H, W)
        f = apply_spatial_to_rgb(f, params)
        frames.append(f)
    video = torch.stack(frames, dim=1)                   # (C, T, crop, crop)

    # -- Photometric: apply per-frame (train only) --
    if is_train and color_augmentor is not None:
        frames = []
        for t in range(T):
            f = video[:, t, :, :]                        # (C, H, W)
            # ColorJitter expects PIL or (C,H,W) float tensor
            f = color_augmentor(f)
            frames.append(f)
        video = torch.stack(frames, dim=1)

    # -- Normalize --
    mean_t = torch.tensor(mean, dtype=video.dtype).view(C, 1, 1, 1)
    std_t  = torch.tensor(std,  dtype=video.dtype).view(C, 1, 1, 1)
    video  = (video - mean_t) / std_t

    return video


# ============================================================
# FLOW AUGMENTATION (geometric only, with corrections)
# ============================================================

def apply_spatial_to_flow(
    flow_x: torch.Tensor,         # (1, T, H, W) or (T, H, W) — horizontal component u
    flow_y: torch.Tensor,         # (1, T, H, W) or (T, H, W) — vertical component v
    params: SpatialAugParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the SAME spatial crop as RGB to flow maps.
    On horizontal flip: negates flow_x (u component reverses direction).
    flow_y is spatially flipped but NOT negated.

    Corrections:
      - hflip:  flow_x = -flip(flow_x),  flow_y = flip(flow_y)
      - crop:   same window, no vector correction needed
    """
    squeeze = flow_x.dim() == 3
    if squeeze:
        flow_x = flow_x.unsqueeze(0)   # (1, T, H, W)
        flow_y = flow_y.unsqueeze(0)

    _, T, H, W = flow_x.shape

    fx_frames, fy_frames = [], []
    for t in range(T):
        fx = flow_x[:, t, :, :]       # (1, H, W)
        fy = flow_y[:, t, :, :]

        # Same crop
        fx = TF.crop(fx, params.top, params.left, params.crop_size, params.crop_size)
        fy = TF.crop(fy, params.top, params.left, params.crop_size, params.crop_size)

        # Horizontal flip with vector correction
        if params.do_hflip:
            fx = -TF.hflip(fx)        # negate u — direction reverses
            fy =  TF.hflip(fy)        # v unchanged in magnitude

        fx_frames.append(fx)
        fy_frames.append(fy)

    flow_x = torch.stack(fx_frames, dim=1)   # (1, T, crop, crop)
    flow_y = torch.stack(fy_frames, dim=1)

    if squeeze:
        flow_x = flow_x.squeeze(0)
        flow_y = flow_y.squeeze(0)

    return flow_x, flow_y


# ============================================================
# MODALITY DROPOUT (training only)
# ============================================================

def apply_modality_dropout(
    z_audio: torch.Tensor,
    z_video: torch.Tensor,
    z_flow:  torch.Tensor,
    p:       float,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Independently zeroes each modality's entire feature vector with probability p.
    Applied only during training.
    Ensures at least one modality always survives to prevent all-zero fusion.
    """
    if not training or p == 0.0:
        return z_audio, z_video, z_flow

    modalities = [z_audio, z_video, z_flow]
    drop_mask  = [random.random() < p for _ in modalities]

    # Safety: if all would be dropped, randomly keep one
    if all(drop_mask):
        keep_idx = random.randint(0, 2)
        drop_mask[keep_idx] = False

    result = []
    for z, drop in zip(modalities, drop_mask):
        if drop:
            result.append(torch.zeros_like(z))
        else:
            result.append(z)

    return result[0], result[1], result[2]
