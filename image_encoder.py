"""
Image encoder for room visual conditioning in the RIR diffusion model.

Extracts multi-scale patch tokens from a frozen DA2 ViT backbone and projects
them to out_dim, ready to be concatenated with the scalar conditioning token
for UNet cross-attention.

Feature extraction modes
------------------------
single   — patch tokens from the final transformer layer only.
last_n   — patch tokens from the last `last_n` layers.
pyramid  — patch tokens from evenly-spaced layers covering the full model
           depth, capturing both geometric structure (early layers) and
           semantic / texture information (late layers).  Default.

Layer combination methods
-------------------------
fuse  (default) — features from all selected layers are concatenated along
                  the channel axis at each patch position, then a single
                  Linear(D*N_layers → out_dim) projects the combined vector:
                      [n_patches, D*N_layers] → [n_patches, out_dim]
                  Produces n_patches tokens; multi-scale information is baked
                  into each spatial token before entering UNet cross-attention.

stack — each selected layer is projected independently with its own
        Linear(D → out_dim), and the resulting sets are stacked:
            [N_layers * n_patches, out_dim]
        Produces more tokens; the UNet can attend to each (layer, position)
        pair independently.

Spatial downsampling (fallback)
--------------------------------
`spatial_pool_factor` applies a strided Conv2d that simultaneously reduces
the spatial grid and re-mixes neighboring patches in a learned way.  Setting
factor=2 halves each spatial dimension (~4× fewer tokens).  Default: None
(full spatial resolution is kept).

Use this when VRAM is constrained. For reference, at batch=4, float16:
  factor=None → ~430 MB KV cache for 1036 patches
  factor=2    → ~108 MB  (259 patches)
  factor=4    → ~27 MB   (63 patches)

Image resolution
----------------
`image_size=(H, W)` controls the resize applied before the backbone.
The default (392, 518) keeps the 4:3 aspect ratio of the rendered 1280×960
images and makes both dimensions divisible by the ViT patch size 14,
giving a 28×37 = 1036-patch spatial grid.
"""

import math
from typing import Literal, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ViT hidden dimensions
_VIT_DIM = {'large': 1024, 'base': 768}

# Number of transformer layers per variant
_VIT_N_LAYERS = {'large': 24, 'base': 12}

# Default pyramid layer indices (1-based: layer k = output after k-th transformer block)
_PYRAMID_LAYERS = {
    'large': [6, 12, 18, 24],
    'base':  [3,  6,  9, 12],
}

# HuggingFace model IDs
_MODEL_IDS = {
    'large': 'depth-anything/Depth-Anything-V2-Large-hf',
    'base':  'depth-anything/Depth-Anything-V2-Base-hf',
}

# Default image size: (H, W) divisible by patch size 14 with ~4:3 aspect ratio.
# Results in a 28×37 = 1036-patch spatial grid.
_DEFAULT_IMAGE_SIZE = (392, 518)


class ImageEncoder(nn.Module):
    """
    Frozen DA2 ViT backbone with configurable multi-layer feature extraction
    and a final linear projection to out_dim.

    Output: [B, T, out_dim] cross-attention tokens ready for the diffusion model.
        fuse  mode: T = n_patches  (spatial resolution preserved)
        stack mode: T = N_layers × n_patches

    Args:
        out_dim:              Output dimension per token.  Must equal the model's
                              cross_attention_dim so tokens can be concatenated
                              with the scalar conditioning token.
        model_variant:        'large' (ViT-L, 24 layers, 1024-dim) or
                              'base'  (ViT-B, 12 layers,  768-dim).
        feature_mode:         'single' | 'last_n' | 'pyramid'  (default 'pyramid')
        last_n:               Layers used when feature_mode='last_n'.
        pyramid_layers:       Explicit 1-based layer indices for the pyramid.
                              None → use default for model_variant.
        layer_combination:    'fuse' | 'stack'  (default 'fuse')
        spatial_pool_factor:  Strided-conv downsampling factor for the spatial
                              grid, or None for full resolution (default None).
        image_size:           (H, W) fed to the backbone; both must be divisible
                              by 14 (ViT patch size).
    """

    def __init__(
        self,
        out_dim: int = 128,
        model_variant: Literal['large', 'base'] = 'large',
        feature_mode: Literal['single', 'last_n', 'pyramid'] = 'pyramid',
        last_n: int = 4,
        pyramid_layers: Optional[List[int]] = None,
        layer_combination: Literal['fuse', 'stack'] = 'fuse',
        spatial_pool_factor: Optional[int] = None,
        image_size: tuple = _DEFAULT_IMAGE_SIZE,
    ):
        super().__init__()

        self.out_dim             = out_dim
        self.feature_mode        = feature_mode
        self.layer_combination   = layer_combination
        self.spatial_pool_factor = spatial_pool_factor
        self.image_size          = image_size

        vit_dim   = _VIT_DIM[model_variant]
        n_layers  = _VIT_N_LAYERS[model_variant]

        # ── backbone ──────────────────────────────────────────────────────────
        from transformers import AutoModelForDepthEstimation
        da2 = AutoModelForDepthEstimation.from_pretrained(_MODEL_IDS[model_variant])
        self.backbone = da2.backbone        # Dinov2Model
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ── layer index selection (1-based; hidden_states[k] = layer k output) ─
        if feature_mode == 'single':
            self._layers = [n_layers]
        elif feature_mode == 'last_n':
            self._layers = list(range(n_layers - last_n + 1, n_layers + 1))
        else:  # 'pyramid'
            self._layers = (pyramid_layers if pyramid_layers is not None
                            else _PYRAMID_LAYERS[model_variant])

        n_selected = len(self._layers)

        # ── optional spatial downsampling (strided conv, learned) ─────────────
        self.spatial_conv = None
        if spatial_pool_factor is not None:
            f = spatial_pool_factor
            self.spatial_conv = nn.Conv2d(
                vit_dim, vit_dim,
                kernel_size=f, stride=f,
                bias=False,
            )

        # ── final projection + normalisation ──────────────────────────────────
        # A LayerNorm after projection normalises token magnitudes before the
        # tokens enter UNet cross-attention.  This is especially important
        # here because the input is a concatenation of features from multiple
        # transformer layers, which can have significantly different scales.
        #
        # fuse:  one projection over the concatenated multi-layer features
        # stack: one projection per selected layer (independent per-layer encoding)
        if layer_combination == 'fuse':
            self.proj = nn.Linear(vit_dim * n_selected, out_dim)
        else:
            self.proj = nn.ModuleList(
                [nn.Linear(vit_dim, out_dim) for _ in range(n_selected)]
            )
        self.norm = nn.LayerNorm(out_dim)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, N_img, 3, H, W]  ImageNet-normalised

        Returns:
            tokens: [B, T, out_dim]
        """
        B, N_img = images.shape[:2]

        # Flatten batch + image axes for the backbone
        x = images.view(B * N_img, *images.shape[2:])          # [B*N, 3, H, W]

        # Forward through backbone, collecting all hidden states.
        # hidden_states is a tuple of length n_layers+1:
        #   [0]  = patch embeddings (before transformer blocks)
        #   [k]  = output after the k-th transformer block  (k = 1 … n_layers)
        # Each tensor: [B*N, 1 + n_patches, D]  (index 0 = CLS token)
        outputs = self.backbone(x, output_hidden_states=True)
        hidden  = outputs.hidden_states

        # Strip CLS token → [B*N, n_patches, D] per layer
        selected = [hidden[k][:, 1:, :] for k in self._layers]

        # ── optional spatial downsampling ─────────────────────────────────────
        if self.spatial_conv is not None:
            selected = [self._apply_spatial_conv(s) for s in selected]

        # ── combine layers ────────────────────────────────────────────────────
        if self.layer_combination == 'fuse':
            # Concatenate along feature dim → single projection per patch
            fused  = torch.cat(selected, dim=-1)        # [B*N, n_patches, D*N_layers]
            tokens = self.proj(fused)                   # [B*N, n_patches, out_dim]
        else:
            # Project each layer independently → stack along patch axis
            projected = [p(s) for p, s in zip(self.proj, selected)]
            tokens    = torch.cat(projected, dim=1)     # [B*N, N_layers*n_patches, out_dim]

        tokens = self.norm(tokens)                      # Normalise token magnitudes

        # Merge N_img back into the token sequence → [B, N_img*T, out_dim]
        T      = tokens.shape[1]
        tokens = tokens.view(B, N_img * T, self.out_dim)
        return tokens

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_spatial_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape patch tokens to 2-D spatial grid, apply strided conv, flatten back."""
        BN, n_patches, D = x.shape
        H_p = self.image_size[0] // 14
        W_p = self.image_size[1] // 14

        x = x.view(BN, H_p, W_p, D).permute(0, 3, 1, 2)   # [BN, D, H_p, W_p]
        x = self.spatial_conv(x)                             # [BN, D, H_p', W_p']
        x = x.permute(0, 2, 3, 1).reshape(BN, -1, D)        # [BN, n_patches', D]
        return x

    # ──────────────────────────────────────────────────────────────────────────
    def n_tokens(self, n_img: int = 1) -> int:
        """Number of cross-attention tokens produced for n_img images per sample."""
        H_p = self.image_size[0] // 14
        W_p = self.image_size[1] // 14

        if self.spatial_conv is not None:
            f   = self.spatial_conv.stride[0]
            H_p = math.ceil(H_p / f)
            W_p = math.ceil(W_p / f)

        n_patches = H_p * W_p
        if self.layer_combination == 'stack':
            n_patches *= len(self._layers)

        return n_img * n_patches
