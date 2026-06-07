"""
pipeline.py
-----------
THE place to add new architectures.

Each pipeline is a self-contained dict with keys:
    build_model  : (cfg) -> nn.Module           — builds the full model
    build_optim  : (model, cfg) -> optimizer    — builds optimizer
    build_sched  : (optimizer, cfg) -> scheduler or None
    build_loss   : (cfg) -> loss_fn

To add a completely new architecture:
    1. Write a new nn.Module anywhere (inline here, or import from a file)
    2. Register it: PIPELINES["my_pipeline"] = { "build_model": ..., ... }
    3. Set cfg.pipeline = "my_pipeline" in config.py or via CLI

Nothing in train.py / test.py needs to change.

--- CLIP MULTIMODAL PIPELINE ---
clip_multimodal replaces cross-entropy with InfoNCE (contrastive) loss.
  • Three frozen encoders (audio ResNet-18, video SlowFast, flow SlowOnly)
    → per-modality projection → concat-fused (1536) → visual_proj (512) → L2-norm
  • Frozen CLIP text encoder encodes all N*D prompts (N classes x 3 domains),
    projects, averages over domains → (N, 512) L2-normalised text embeddings
  • InfoNCE loss: bring fused embedding close to GT class text embedding,
    push away from the other N-1 class text embeddings.
  • At inference: argmax of cosine similarities → predicted class.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau, LambdaLR
)

from config import Config


# ============================================================
# SHARED UTILITIES
# ============================================================

def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def count_params(module: nn.Module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def pack_slowfast(video: torch.Tensor, alpha: int = 4):
    return [video[:, :, ::alpha, :, :], video]


# ============================================================
# BUILDING BLOCKS
# ============================================================

class ProjectionMLP(nn.Module):
    """Linear → optional BN → optional ReLU projection."""
    def __init__(self, in_dim: int, out_dim: int, use_bn=True, use_relu=True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if use_bn:   layers.append(nn.BatchNorm1d(out_dim))
        if use_relu: layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def build_audio_encoder(cfg: Config) -> nn.Module:
    """ResNet-18 audio encoder (VGGSound ckpt). Output: (B, 512)."""
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Identity()
    state = torch.load(cfg.paths.audio_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    freeze(model)
    return model


def build_video_encoder(cfg: Config, ckpt_key="video_ckpt") -> nn.Module:
    """SlowFast-R101 (Kinetics ckpt). Output: (B, 2304)."""
    from pytorchvideo.models.hub import slowfast_r101
    model = slowfast_r101(pretrained=False)
    model.blocks[-1].proj = nn.Identity()
    state = torch.load(getattr(cfg.paths, ckpt_key), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    freeze(model)
    return model


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    t = cfg.training
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters.")
    kwargs = dict(lr=t.lr, weight_decay=t.weight_decay)
    name = t.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, betas=t.betas, eps=t.eps, **kwargs)
    elif name == "adam":
        return torch.optim.Adam(params, betas=t.betas, eps=t.eps, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(params, momentum=t.momentum, nesterov=t.nesterov, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: Config, num_epochs: int):
    t = cfg.training
    name = t.scheduler.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=t.eta_min)
    elif name == "cosine_warmup":
        warmup = t.warmup_epochs
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(num_epochs - warmup, 1)
            return t.eta_min + 0.5 * (1.0 - t.eta_min) * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)
    elif name == "step":
        return StepLR(optimizer, step_size=t.step_size, gamma=t.gamma)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=t.plateau_patience, factor=t.plateau_factor)
    elif name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def build_loss(cfg: Config) -> nn.Module:
    if cfg.pipeline in ("clip_multimodal",):
        return InfoNCELoss()
    return nn.CrossEntropyLoss(label_smoothing=0.1)


# ============================================================
# PIPELINE 1: base_multimodal
# ============================================================

class BaseMultimodalModel(nn.Module):
    def __init__(
        self,
        audio_encoder, video_encoder, flow_encoder,
        common_dim:       int   = 512,
        fusion:           str   = "sum",
        num_classes:      int   = 7,
        hidden_dim:       int   = None,
        dropout_p:        float = 0.3,
        modal_dropout_p:  float = 0.15,
    ):
        super().__init__()
        self.audio_enc = audio_encoder
        self.video_enc = video_encoder
        self.flow_enc  = flow_encoder

        self.proj_audio = ProjectionMLP(512,  common_dim)
        self.proj_video = ProjectionMLP(2304, common_dim)
        self.proj_flow  = ProjectionMLP(2304, common_dim)

        self.fusion_method  = fusion
        self.modal_drop_p   = modal_dropout_p
        fused_dim = common_dim if fusion == "sum" else common_dim * 3

        self.dropout = nn.Dropout(p=dropout_p)

        if hidden_dim is None:
            self.classifier = nn.Linear(fused_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p), nn.Linear(hidden_dim, num_classes)
            )

        total, trainable = count_params(self)
        print(f"[BaseMultimodalModel] fusion={fusion} fused_dim={fused_dim} "
              f"total={total:,} trainable={trainable:,}")

    def _fuse(self, z_audio, z_video, z_flow):
        if self.fusion_method == "sum":
            return (z_audio + z_video + z_flow) / 3.0
        return torch.cat([z_audio, z_video, z_flow], dim=1)

    def _modal_dropout(self, z_audio, z_video, z_flow):
        import random
        if not self.training or self.modal_drop_p == 0.0:
            return z_audio, z_video, z_flow
        mods  = [z_audio, z_video, z_flow]
        drops = [random.random() < self.modal_drop_p for _ in mods]
        if all(drops):
            drops[random.randint(0, 2)] = False
        return tuple(torch.zeros_like(z) if d else z for z, d in zip(mods, drops))

    def forward(self, audio, video, flow_x, flow_y):
        f_audio = self.audio_enc(audio)
        f_video = self.video_enc(pack_slowfast(video))
        flow_avg = (flow_x + flow_y) / 2.0
        slow, fast = pack_slowfast(flow_avg)
        f_flow  = self.flow_enc([slow, torch.zeros_like(fast)])

        z_audio, z_video, z_flow = (
            self.proj_audio(f_audio),
            self.proj_video(f_video),
            self.proj_flow(f_flow),
        )

        z_audio, z_video, z_flow = self._modal_dropout(z_audio, z_video, z_flow)
        z = self._fuse(z_audio, z_video, z_flow)
        return self.classifier(self.dropout(z))


def _build_base_multimodal(cfg: Config) -> nn.Module:
    return BaseMultimodalModel(
        audio_encoder    = build_audio_encoder(cfg),
        video_encoder    = build_video_encoder(cfg, "video_ckpt"),
        flow_encoder     = build_video_encoder(cfg, "flow_ckpt"),
        common_dim       = 512,
        fusion           = "sum",
        num_classes      = 7,
        hidden_dim       = None,
        dropout_p        = 0.3,
        modal_dropout_p  = 0.15,
    )


# ============================================================
# PIPELINE 2: concat_multimodal
# ============================================================

def _build_concat_multimodal(cfg: Config) -> nn.Module:
    return BaseMultimodalModel(
        audio_encoder    = build_audio_encoder(cfg),
        video_encoder    = build_video_encoder(cfg, "video_ckpt"),
        flow_encoder     = build_video_encoder(cfg, "flow_ckpt"),
        common_dim       = 512,
        fusion           = "concat",
        num_classes      = 7,
        hidden_dim       = 512,
        dropout_p        = 0.3,
        modal_dropout_p  = 0.15,
    )


# ============================================================
# PIPELINE 3: audio_only
# ============================================================

class AudioOnlyModel(nn.Module):
    def __init__(self, audio_encoder, num_classes=7, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        self.encoder    = audio_encoder
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p), nn.Linear(hidden_dim, num_classes)
        )
        total, trainable = count_params(self)
        print(f"[AudioOnlyModel] total={total:,} trainable={trainable:,}")

    def forward(self, audio, video=None, flow_x=None, flow_y=None):
        return self.classifier(self.encoder(audio))


def _build_audio_only(cfg: Config) -> nn.Module:
    return AudioOnlyModel(build_audio_encoder(cfg), num_classes=7)


# ============================================================
# PIPELINE 4: clip_multimodal
#   concat fusion (1536→512) + InfoNCE against CLIP text embeddings
#   Domain-ensemble: 3 prompts per class (person/animal/cartoon),
#   averaged to one robust text embedding per class.
# ============================================================

class InfoNCELoss(nn.Module):
    """Cross-entropy over cosine-similarity logits — identical call signature to nn.CrossEntropyLoss."""
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


def build_clip_text_encoder(model_name: str):
    from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel
    try:
        tokenizer    = CLIPTokenizer.from_pretrained(model_name)
        text_encoder = CLIPTextModel.from_pretrained(model_name)
        use_pooler   = hasattr(text_encoder.config, "projection_dim")
    except Exception:
        tokenizer    = AutoTokenizer.from_pretrained(model_name)
        text_encoder = AutoModel.from_pretrained(model_name).text_model
        use_pooler   = False
    text_dim = text_encoder.config.hidden_size
    return text_encoder, tokenizer, text_dim, use_pooler


class CLIPTextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder, tokenizer, text_dim: int,
                 use_pooler: bool, freeze: bool = True):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer    = tokenizer
        self.text_dim     = text_dim
        self.use_pooler   = use_pooler
        if freeze:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def encode_prompts(self, prompts: list, device: torch.device) -> torch.Tensor:
        """Tokenise and encode prompts → L2-normalised (N, text_dim)."""
        inputs = self.tokenizer(
            prompts, padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.text_encoder(**inputs)
        if self.use_pooler and hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            mask   = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = out.last_hidden_state
            emb    = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=-1)


class CLIPMultimodalModel(nn.Module):
    """
    Visual path:
        audio_enc (512) + video_enc (2304) + flow_enc (2304)
        → proj_audio/video/flow (each → 512)
        → concat (1536)
        → visual_proj Linear(1536→512) + BN
        → L2-norm → z_vis (B, 512)

    Text path (frozen CLIP, runs once per forward):
        For each of N classes, build 3 prompts (person / animal / cartoon character).
        Encode all N*3 prompts → project → average over 3 domains → (N, 512)
        → L2-norm → z_txt (N, 512)

    Loss:
        logits = z_vis @ z_txt.T / τ    (B, N)
        InfoNCE = CrossEntropy(logits, gt_labels)

    Inference:
        predicted = argmax(logits, dim=1)
    """
    def __init__(
        self,
        audio_encoder,
        video_encoder,
        flow_encoder,
        text_encoder_wrapper: CLIPTextEncoderWrapper,
        action_classes:   list,
        common_dim:       int   = 512,
        proj_dim:         int   = 512,
        dropout_p:        float = 0.3,
        modal_dropout_p:  float = 0.15,
        init_temp:        float = 0.07,
        learn_temp:       bool  = True,
    ):
        super().__init__()

        # Visual encoders (all frozen)
        self.audio_enc = audio_encoder
        self.video_enc = video_encoder
        self.flow_enc  = flow_encoder

        # Per-modality projections
        self.proj_audio = ProjectionMLP(512,  common_dim)
        self.proj_video = ProjectionMLP(2304, common_dim)
        self.proj_flow  = ProjectionMLP(2304, common_dim)

        # Concat fusion → visual projection
        fused_dim = common_dim * 3   # 1536
        self.visual_proj = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

        # Text encoder + projection
        self.text_enc_wrapper = text_encoder_wrapper
        self.text_proj = nn.Sequential(
            nn.Linear(text_encoder_wrapper.text_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

        # Learnable temperature
        log_tau = math.log(init_temp)
        if learn_temp:
            self.log_tau = nn.Parameter(torch.tensor(log_tau))
        else:
            self.register_buffer("log_tau", torch.tensor(log_tau))

        # Dropout + modal dropout
        self.dropout      = nn.Dropout(p=dropout_p)
        self.modal_drop_p = modal_dropout_p

        # Domain-ensemble prompts
        # 3 prompts per class (person / animal / cartoon), flat list of N*3
        self.action_classes = action_classes
        self.domain_subjects = ["a person", "an animal", "a cartoon character"]
        self.prompts_flat = [
            f"a video of {subject} {action}"
            for action in action_classes
            for subject in self.domain_subjects
        ]

        total, trainable = count_params(self)
        print(f"[CLIPMultimodalModel] fusion=concat fused_dim={fused_dim} proj_dim={proj_dim} "
              f"num_classes={len(action_classes)} domains={len(self.domain_subjects)} "
              f"total={total:,} trainable={trainable:,}")

    def _modal_dropout(self, z_audio, z_video, z_flow):
        import random
        if not self.training or self.modal_drop_p == 0.0:
            return z_audio, z_video, z_flow
        mods  = [z_audio, z_video, z_flow]
        drops = [random.random() < self.modal_drop_p for _ in mods]
        if all(drops):
            drops[random.randint(0, 2)] = False
        return tuple(torch.zeros_like(z) if d else z for z, d in zip(mods, drops))

    def _encode_visual(self, audio, video, flow_x, flow_y) -> torch.Tensor:
        """Returns L2-normalised visual embedding: (B, proj_dim)."""
        f_audio = self.audio_enc(audio)
        f_video = self.video_enc(pack_slowfast(video))
        flow_avg = (flow_x + flow_y) / 2.0
        slow, fast = pack_slowfast(flow_avg)
        f_flow  = self.flow_enc([slow, torch.zeros_like(fast)])

        z_audio = self.proj_audio(f_audio)
        z_video = self.proj_video(f_video)
        z_flow  = self.proj_flow(f_flow)

        z_audio, z_video, z_flow = self._modal_dropout(z_audio, z_video, z_flow)

        z_fused = torch.cat([z_audio, z_video, z_flow], dim=1)   # (B, 1536)
        z_vis   = self.visual_proj(self.dropout(z_fused))         # (B, 512)
        return F.normalize(z_vis, p=2, dim=-1)

    def _encode_text(self, device: torch.device) -> torch.Tensor:
        """
        Encodes N*3 domain-ensemble prompts, projects, averages over 3 domains.
        Returns L2-normalised text embeddings: (N, proj_dim).
        """
        N = len(self.action_classes)
        D = len(self.domain_subjects)

        # (N*D, text_dim)
        t_raw  = self.text_enc_wrapper.encode_prompts(self.prompts_flat, device)
        # (N*D, proj_dim)
        t_proj = self.text_proj(t_raw)
        # (N, D, proj_dim) → average over D → (N, proj_dim)
        t_proj = t_proj.view(N, D, -1).mean(dim=1)
        return F.normalize(t_proj, p=2, dim=-1)

    def forward(self, audio, video, flow_x, flow_y) -> torch.Tensor:
        """Returns similarity logits (B, N) = z_vis @ z_txt.T / τ."""
        device = audio.device
        z_vis  = self._encode_visual(audio, video, flow_x, flow_y)  # (B, 512)
        z_txt  = self._encode_text(device)                           # (N, 512)
        tau    = self.log_tau.exp().clamp(min=1e-4, max=100.0)
        return (z_vis @ z_txt.T) / tau                               # (B, N)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


def _build_clip_multimodal(cfg: Config) -> nn.Module:
    cc = cfg.clip
    print(f"[clip_multimodal] Loading text encoder: {cc.model_name}")
    text_encoder, tokenizer, text_dim, use_pooler = build_clip_text_encoder(cc.model_name)
    text_wrapper = CLIPTextEncoderWrapper(
        text_encoder = text_encoder,
        tokenizer    = tokenizer,
        text_dim     = text_dim,
        use_pooler   = use_pooler,
        freeze       = cc.freeze_text_encoder,
    )
    return CLIPMultimodalModel(
        audio_encoder        = build_audio_encoder(cfg),
        video_encoder        = build_video_encoder(cfg, "video_ckpt"),
        flow_encoder         = build_video_encoder(cfg, "flow_ckpt"),
        text_encoder_wrapper = text_wrapper,
        action_classes       = cc.action_classes,
        common_dim           = 512,
        proj_dim             = cc.proj_dim,
        dropout_p            = 0.3,
        modal_dropout_p      = 0.15,
        init_temp            = cc.init_temp,
        learn_temp           = cc.learn_temp,
    )


# ============================================================
# PIPELINE REGISTRY
# ============================================================

PIPELINES: dict = {
    "base_multimodal":   _build_base_multimodal,
    "concat_multimodal": _build_concat_multimodal,
    "audio_only":        _build_audio_only,
    "clip_multimodal":   _build_clip_multimodal,
}


def build_pipeline(cfg: Config) -> nn.Module:
    name = cfg.pipeline
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: '{name}'. Available: {list(PIPELINES)}")
    return PIPELINES[name](cfg)