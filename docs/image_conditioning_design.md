# Image Conditioning Design

## Background

The model previously conditioned on a scalar vector only:
`[room_dims(3), mic_loc(3), speaker_loc(3), rt60(1)]` → MLP encoder → `[B, 1, D]` → UNet cross-attention.

This document describes the image conditioning extension, which adds per-pair RGB images
as additional cross-attention tokens while keeping the scalar-only (GTU) path completely unchanged.

---

## Full Conditioning Flow (SoundSpaces)

```
Input images  [B, N_img, 3, H, W]
      │
      ▼  ═══════════════ ImageEncoder ════════════════════════════════════════
      │
      ├─ [1] Backbone forward (frozen DA3 ViT)
      │       → hidden states at selected pyramid layers
      │
      ├─ [2] Strip CLS token, keep patch tokens
      │       → N_L × [B·N_img, P, D_vit]
      │
      ├─ [3] Fuse layers (concat channel-wise) + Linear projection
      │       → [B·N_img, P, D]
      │
      └─ [4] LayerNorm
              → image_tokens  [B, N_img·P, D]
      │
      ▼  ═════════════════════════════════════════════════════════════════════

Scalar params  [B, C_s]
      │
      ▼
 Conditioning MLP encoder  →  scalar_token  [B, 1, D]

      │                              │
      └──────────┬───────────────────┘
                 │
                 ▼  ── FusionBlock (when image_fusion='cross_attn') ──────────
                 │
                 ├─ Pre-LN on scalar_token and image_tokens
                 ├─ scalar_token (query) attends to image_tokens (key/value)
                 └─ Residual: fused_scalar = scalar_token + attn_out  [B, 1, D]
                 │
                 ▼  ─────────────────────────────────────────────────────────

      cat([fused_scalar, image_tokens], dim=1)
        →  cond_seq  [B, 1 + N_img·P, D]

                 ▼
           UNet cross-attention
      (spectrogram patches attend to all 1 + N_img·P conditioning tokens)
```

**Shape reference:**
- `P` = number of patch tokens per image = `(H/14) × (W/14)` (e.g. 392×518 input → 28×37 = 1036)
- `D_vit` = backbone hidden dim (1024 for ViT-L, 768 for ViT-B)
- `N_L` = number of selected layers (default 4 for pyramid)
- `D` = `out_dim` = `cross_attention_dim` (default 128; see experiment note below)
- `C_s` = scalar conditioning dim (9 for SoundSpaces, 10 for GTU with RT60)

**DA3 (Depth Anything 3):** a monocular depth estimation model built on DINOv2.
Its ViT encoder is geometry-aware and used here as a frozen feature extractor;
the depth estimation head is discarded.

For the GTU dataset, the flow stops after the MLP encoder:
`scalar_token [B, 1, D]` → UNet cross-attention directly.

---

## Dataset Differences

| | GTU | SoundSpaces |
|---|---|---|
| Scalar dim `C_s` | 10 (includes RT60) | 9 (no RT60) |
| Images | None | N_img per RIR (default 1: receiver→source RGB) |
| Batch key `'rt60'` | Present | Absent |
| Batch key `'images'` | Absent | Present (when `image_root` is set) |
| Collate function | `dataset_type='gtu'` | `dataset_type='soundspaces'` |

---

## ImageEncoder Configuration

| Parameter | Options | Default | Effect |
|---|---|---|---|
| `model_variant` | `'large'` / `'base'` | `'large'` | ViT-L (24 layers, D_vit=1024) or ViT-B (12 layers, D_vit=768) |
| `feature_mode` | `'single'` / `'last_n'` / `'pyramid'` | `'pyramid'` | Which backbone layers to extract |
| `last_n` | int | 4 | Layers used when `feature_mode='last_n'` |
| `pyramid_layers` | list of ints | ViT-L: [6,12,18,24] | Explicit layer indices (1-based) |
| `layer_combination` | `'fuse'` / `'stack'` | `'fuse'` | How to combine selected layers (see below) |
| `spatial_pool_factor` | int or None | None | Strided-conv downsampling; use when VRAM is tight |
| `image_size` | (H, W) | (392, 518) | Resize resolution fed to backbone (both dims must be divisible by 14) |

### `feature_mode` explained

- **`single`** — patch tokens from the final transformer layer only.
- **`last_n`** — patch tokens from the last N layers.
- **`pyramid`** — patch tokens from evenly-spaced layers (default). Captures both early
  geometric structure (low layers) and semantic/texture information (high layers).

### `layer_combination` explained

- **`fuse`** (default) — features from all N_L selected layers are concatenated channel-wise
  at each patch position → `[P, D_vit × N_L]`, then projected with a single
  `Linear(D_vit × N_L → D)`. Produces P tokens per image. Multi-scale information
  is baked into each spatial token.

- **`stack`** — each layer is projected independently with its own `Linear(D_vit → D)`
  and the results stacked → `N_L × P` tokens per image. The UNet can attend to each
  (layer, position) pair independently, at the cost of more tokens.

### Spatial downsampling (VRAM trade-off)

Default: no downsampling — full P patches preserved for maximum geometric precision.
`spatial_pool_factor=N` applies a learned strided Conv2d, reducing the spatial grid by N
in each dimension (e.g. N=2: 28×37 → 14×19, reducing P by ~4×).

| `spatial_pool_factor` | Patches P (392×518 input) | Approx. KV cache (batch=4, fp16) |
|---|---|---|
| None (default) | 1036 | ~430 MB |
| 2 | ~266 | ~110 MB |
| 4 | ~70 | ~29 MB |

---

## FusionBlock Configuration

Controlled by `image_fusion` in `RIRDiffusionModel`:

- **`'cross_attn'`** (default) — inserts `FusionBlock`: the scalar token queries image
  tokens before both enter UNet cross-attention. Uses Pre-LN on both inputs and a
  residual connection to ensure spatial coordinates are preserved.

- **`'concat'`** — skips FusionBlock entirely. Scalar token and image tokens are
  concatenated directly and passed to the UNet. Simpler, fewer parameters.
  Use as ablation baseline.

---

## Key Design Notes

**Why residual in FusionBlock?**
The scalar token carries precise floating-point room geometry coordinates (metres).
Without a residual, cross-attention can suppress these in favour of visual features.
The residual guarantees the spatial coordinates survive fusion unchanged.

**Why Pre-LN in FusionBlock?**
The scalar MLP output has no prior normalisation. Pre-LN on both inputs stabilises
attention scores when the two modalities have different activation scales.

**Why LayerNorm after ImageEncoder projection?**
The projection input concatenates features from N_L transformer layers with different
activation scales. LN normalises token magnitudes so cross-attention scores are driven
by content rather than inter-layer scale differences.

---

## ⚠️ Experiment Recommendation: Raise `out_dim`

The current `out_dim=128` (= `cross_attention_dim` = `D` in this doc) was inherited from
the scalar conditioning MLP's output size. For up to N_img × P image patch tokens carrying
multi-layer visual information, **128 dimensions is a significant bottleneck**.
Models like CLIP and DINOv2 use 512–1024-dim embeddings for comparable visual inputs.

Raising `out_dim` requires raising `cross_attention_dim` in the UNet to match (both scalar
and image tokens must share the same dimension). Recommended values to explore:

| `out_dim` / `cross_attention_dim` | Notes |
|---|---|
| 128 (current) | Minimal memory; likely under-capacity for image features |
| 256 | Good first step; doubles capacity with modest memory increase |
| 512 | Closer to standard vision-language embedding dims |

This is one of the highest-impact hyperparameters to ablate once baseline runs are established.
