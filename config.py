"""
config.py
---------
Data, training, and path settings only.
Architecture is fully defined inside each pipeline (see pipeline.py).
To change architecture: set cfg.pipeline = "your_pipeline_name".
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ============================================================
# PATHS
# ============================================================

@dataclass
class PathConfig:
    audio_dir:         str = "/datasets/HAC/{domain}/audio"
    video_dir:         str = "/datasets/HAC/{domain}/videos"
    flow_dir:          str = "/datasets/HAC/{domain}/flow"
    train_csv:         str = "/home/sameer/workspace/Sameer/Files/Annotations/HAC_train_only_{domain}.csv"
    test_csv:          str = "/home/sameer/workspace/Sameer/Files/Annotations/HAC_test_only_{domain}.csv"
    audio_ckpt:        str = "/home/sameer/workspace/Sameer/checkpoints/vggaudio/vggsound_netvlad.pth"
    video_ckpt:        str = "/home/sameer/workspace/Sameer/checkpoints/Slowfast_chkptk400/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth"
    flow_ckpt:         str = "/home/sameer/workspace/Sameer/checkpoints/slowonly_chckptk400/slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb_20220901-e6281431.pth"
    frames_cache_dir:  str = "/home/sameer/workspace/Sameer/Files/Expt-Aug+Dropout/frames_cache"
    feature_cache_dir: str = "/home/sameer/workspace/Sameer/Files/Expt-Aug+Dropout/features_cache"
    output_dir:        str = "./outputs/cartoon"


# ============================================================
# DOMAINS
# ============================================================

@dataclass
class DomainConfig:
    all_domains:    List[str] = field(default_factory=lambda: ["human", "animal", "cartoon"])
    train_domains:  List[str] = field(default_factory=lambda: ["cartoon"])
    test_domains:   List[str] = field(default_factory=lambda: ["human", "animal", "cartoon"])


# ============================================================
# AUDIO (data-loading only)
# ============================================================

@dataclass
class AudioConfig:
    sample_rate:   int   = 16000
    clip_duration: float = 10.0
    n_fft:         int   = 512
    hop_length:    int   = 160
    win_length:    int   = 512
    augment:       bool  = True
    # Augmentation probabilities
    aug_gaussian_noise_p:    float = 0.4
    aug_gaussian_min:        float = 0.001
    aug_gaussian_max:        float = 0.015
    aug_time_stretch_p:      float = 0.4
    aug_time_stretch_min:    float = 0.9
    aug_time_stretch_max:    float = 1.1
    aug_pitch_shift_p:       float = 0.3
    aug_pitch_min_semitones: float = -3.0
    aug_pitch_max_semitones: float = 3.0
    aug_shift_p:             float = 0.3
    aug_shift_min:           float = -0.2
    aug_shift_max:           float = 0.2
    aug_gain_p:              float = 0.4
    aug_gain_min_db:         float = -6.0
    aug_gain_max_db:         float = 6.0


# ============================================================
# VIDEO / FLOW (data-loading only)
# ============================================================

@dataclass
class VideoConfig:
    num_frames:  int        = 32
    crop_size:   int        = 224
    scale_size:  int        = 256
    mean:        List[float] = field(default_factory=lambda: [0.45, 0.45, 0.45])
    std:         List[float] = field(default_factory=lambda: [0.225, 0.225, 0.225])
    augment_color:           bool  = True
    color_jitter_brightness: float = 0.4
    color_jitter_contrast:   float = 0.4
    color_jitter_saturation: float = 0.2
    color_jitter_hue:        float = 0.1
    color_jitter_p:          float = 0.8
    grayscale_p:             float = 0.1
    augment_spatial:         bool  = True
    random_hflip_p:          float = 0.5


# ============================================================
# CLIP / TEXT ENCODER
# ============================================================

@dataclass
class CLIPConfig:
    # HuggingFace model name for the text encoder.
    # Options (best quality → most VRAM):
    #   "openai/clip-vit-large-patch14"   (CLIP ViT-L/14,  text dim=768)
    #   "openai/clip-vit-base-patch32"    (CLIP ViT-B/32,  text dim=512)
    #   "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"  (OpenCLIP ViT-H, text dim=1024)
    #   "google/siglip-large-patch16-384" (SigLIP ViT-L,  text dim=1024)
    model_name: str = "openai/clip-vit-large-patch14"

    # Prompt template — {action} is replaced with each class name.
    # CLIP zero-shot performance is sensitive to prompt wording.
    prompt_template: str = "a video of {action}"

    # The 7 HAC action classes in label order (index = class id).
    # Adjust names to match your dataset's vocabulary exactly.
    action_classes: List[str] = field(default_factory=lambda: [
        "sleeping",
        "watching tv",
        "eating",
        "drinking",
        "running",
        "swimming",
        "opening door",
    ])

    # Temperature parameter τ for InfoNCE (learnable by default via log_temp).
    # Set init_temp to the starting value of exp(log_τ).
    init_temp: float = 0.07

    # Whether to make τ a learnable parameter (recommended: True).
    learn_temp: bool = True

    # Projection dimension — fused and text embeddings are both projected here.
    proj_dim: int = 512

    # Whether to freeze the text encoder weights entirely.
    freeze_text_encoder: bool = True


# ============================================================
# TRAINING
# ============================================================

@dataclass
class TrainingConfig:
    device:      str  = "cuda:1"
    epochs:      int  = 140
    batch_size:  int  = 24
    num_workers: int  = 4
    seed:        int  = 42
    pin_memory:  bool = True
    top_k:       int  = 5
    eval_every:  int  = 5
    save_best:   bool = True
    save_every:  int  = 20
    # Optimizer
    optimizer:        str   = "adamw"
    lr:               float = 1e-3
    weight_decay:     float = 1e-3
    betas:            tuple = (0.9, 0.999)
    eps:              float = 1e-8
    momentum:         float = 0.9
    nesterov:         bool  = True
    # Scheduler
    scheduler:        str   = "cosine_warmup"
    eta_min:          float = 1e-6
    warmup_epochs:    int   = 10
    step_size:        int   = 50
    gamma:            float = 0.5
    plateau_patience: int   = 10
    plateau_factor:   float = 0.5


# ============================================================
# MASTER CONFIG
# ============================================================

@dataclass
class Config:
    pipeline:  str           = "clip_multimodal"   # ← base_multimodal|clip_multimodal (concat fusion)
    paths:     PathConfig    = field(default_factory=PathConfig)
    domains:   DomainConfig  = field(default_factory=DomainConfig)
    audio:     AudioConfig   = field(default_factory=AudioConfig)
    video:     VideoConfig   = field(default_factory=VideoConfig)
    training:  TrainingConfig = field(default_factory=TrainingConfig)
    clip:      CLIPConfig    = field(default_factory=CLIPConfig)


# ============================================================
# PRESETS  (override cfg fields as needed per experiment)
# ============================================================

def get_config(pipeline: str = "clip_multimodal", **overrides) -> Config:
    cfg = Config(pipeline=pipeline)
    for k, v in overrides.items():
        # support dotted keys: "training.lr" → cfg.training.lr = v
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg
 