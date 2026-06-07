# clip_multimodal - CLIP-guided Contrastive Action Recognition

## What changed (and what didn't)

| File | Status | What changed |
|------|--------|--------------|
| `augmentations.py` | **unchanged** | - |
| `dataloader.py` | **unchanged** | - |
| `metrics.py` | **unchanged** | - |
| `preprocess_frames.py` | **unchanged** | - |
| `config.py` | **+CLIPConfig** | New `CLIPConfig` dataclass added; `Config` gains a `clip` field |
| `pipeline.py` | **+clip_multimodal** | `CLIPTextEncoderWrapper`, `CLIPMultimodalModel`, `InfoNCELoss`, `_build_clip_multimodal` added; existing pipelines untouched |
| `train.py` | **minor** | Temperature logging added; no structural change |
| `test.py` | **minor** | Diagnostic print for clip pipeline; no structural change |

---

## Architecture: `clip_multimodal`

```
Audio  ──► ResNet-18 ──► proj_audio (512→512)  ─┐
Video  ──► SlowFast   ──► proj_video (2304→512) ─┼──► sum-fuse ──► visual_proj ──► L2-norm ──► z_vis (B, proj_dim)
Flow   ──► SlowOnly   ──► proj_flow  (2304→512) ─┘

class_names ──► CLIP text encoder (frozen) ──► text_proj ──► L2-norm ──► z_txt (N, proj_dim)

logits[i,j] = ( z_vis[i] · z_txt[j] ) / τ          shape: (B, N)

InfoNCE loss = CrossEntropy(logits, gt_labels)       ← brings z_vis[i] close to z_txt[gt]
                                                        and pushes it away from all other N-1 classes
```

### Key design decisions

**Frozen encoders, trainable projections.**
The three A/V/F encoders and the CLIP text encoder are all frozen.
Only `proj_audio`, `proj_video`, `proj_flow`, `visual_proj`, `text_proj`, and `log_τ` are trained.
This keeps the memory footprint and training time nearly identical to the base pipeline.

**Shared embedding space.**
Both visual and text embeddings are projected to the same `proj_dim` (default 512) and L2-normalised before the dot product, exactly as in CLIP.

**Learnable temperature τ.**
`log_τ` is stored and optimised as `nn.Parameter`. Starting from τ=0.07 (CLIP's value), the model learns how sharply to peak the distribution. Clipped to `[1e-4, 100]` for stability.

**InfoNCE = per-sample softmax cross-entropy.**
For each sample `i` in the batch the loss is:

```
L_i = -log [ exp(z_vis_i · z_txt_{y_i} / τ) / Σ_j exp(z_vis_i · z_txt_j / τ) ]
```

The negatives are all N–1 other *class text embeddings* (not other samples in the batch), so the loss is well-defined even for batch size 1 and doesn't suffer from batch-composition variance.

**Inference is zero-shot style.**
At test time `model.forward()` returns the same `(B, N)` logit matrix.
`argmax` over dim=1 gives the predicted class - identical to every other pipeline.
`metrics.py` and `test.py` require no changes.

---

## Quick start

### 1. Install the one new dependency

```bash
pip install transformers>=4.35.0
```

### 2. Update `action_classes` in `config.py`

Open `config.py` and set `CLIPConfig.action_classes` to match your 7 HAC labels **in label-index order** (index 0 = class 0, etc.):

```python
action_classes: List[str] = field(default_factory=lambda: [
    "clapping",    # label 0
    "waving",      # label 1
    "pointing",    # label 2
    "jumping",     # label 3
    "running",     # label 4
    "crawling",    # label 5
    "falling",     # label 6
])
```

### 3. Train

```bash
python train.py --pipeline clip_multimodal
```

With overrides:

```bash
python train.py --pipeline clip_multimodal \
    --set clip.model_name=openai/clip-vit-base-patch32 \
    --set clip.proj_dim=256 \
    --set training.lr=5e-4
```

### 4. Test / evaluate

```bash
python test.py --checkpoint outputs/best_clip_multimodal_human.pt
python test.py --checkpoint outputs/best_clip_multimodal_human.pt --domain cartoon
```

---

## Choosing a text encoder

All options are pulled automatically from HuggingFace on first run.

| Model name | Text dim | Quality | VRAM (text enc) |
|------------|----------|---------|-----------------|
| `openai/clip-vit-base-patch32` | 512 | Good | ~430 MB |
| `openai/clip-vit-large-patch14` | 768 | **Better** ← default | ~830 MB |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 1024 | Best open | ~1.6 GB |
| `google/siglip-large-patch16-384` | 1024 | Best overall | ~1.6 GB |

Since the text encoder is **frozen and shared across the whole dataset**, its VRAM cost is fixed and small. The text forward pass encodes only 7 class prompts per forward call - negligible compute.

---

## Prompt template

The default template is:

```
"a video of a person performing {action}"
```

You can change it via config:

```bash
--set clip.prompt_template="an action of {action} performed in a video"
```

CLIP is sensitive to prompt phrasing. For cross-domain generalisation (human → animal/cartoon), more neutral templates often work better.

---

## Differences vs. vanilla CLIP image-text contrastive training

| CLIP original | This pipeline |
|---------------|---------------|
| Image encoder (ViT/ResNet) | **3-stream A/V/F encoder** with sum-fusion |
| Contrastive over (image, text) *pairs in the batch* | Contrastive over *visual embedding vs. all N class texts* |
| Both encoders trained | **All encoders frozen**, only projections trained |
| N = batch size (typically 32k–65k) | N = 7 (number of action classes) |
| Symmetric loss (image→text + text→image) | Asymmetric: only visual→text direction |

The per-class-text negative formulation is closer to zero-shot CLIP evaluation than to the original contrastive pre-training, but it's applied *during fine-tuning* with ground-truth labels - giving the supervised signal while keeping the structured text embedding space.

---

## Resuming / switching pipelines

Checkpoints store `cfg` so the correct pipeline is always restored:

```bash
# Resume a clip_multimodal run
python train.py --resume outputs/ckpt_epoch020_human.pt --epochs 200

# Evaluate with the saved config automatically
python test.py --checkpoint outputs/best_clip_multimodal_human.pt
```

To switch from `base_multimodal` to `clip_multimodal` mid-project, start fresh - the model weights are not compatible because the classifier head is replaced by projection heads.
