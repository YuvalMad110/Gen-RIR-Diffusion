import os
import time
import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from utils.misc import save_metric, get_timestamped_logdir

from tqdm import tqdm
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.models.unets import UNet2DConditionModel
from typing import List, Tuple, Optional, Union

# ToDo:
#     - set torch.nn.Module as superclass ?
#     - use UNet2DConditionModel with the block_out_channels=(32, 64, 128, 128) (duplicate first/last block from up_block_types and down_block_types)
#     - try transformer based model
#     - try different scheduler
#     - understand whether a conditioning encoder is necessary
#     - use RT60(freq)


class FusionBlock(nn.Module):
    """
    Optional cross-attention fusion between the scalar conditioning token and
    image patch tokens, applied before UNet cross-attention.

    The scalar token (query) attends to image tokens (key/value), so the
    spatial conditioning can selectively absorb room-relevant visual cues.
    The result is concatenated with the image tokens to form the full
    conditioning sequence for the UNet.

    Design choices:
      Pre-LN  — both inputs are layer-normalised before attention so that
                neither scale differences between modalities nor the lack of
                prior normalisation on the scalar MLP output destabilise
                the attention scores.
      Residual — the original scalar_token is added back after attention,
                 ensuring that the precise spatial coordinates (room dims,
                 mic/speaker positions) are never washed out by visual
                 features.

    This block is intentionally separate so it can be toggled for ablation
    studies.  Set image_fusion='concat' in RIRDiffusionModel to bypass it
    and fall back to simple concatenation.

    Args:
        dim:       Feature dimension (must equal cross_attention_dim).
        num_heads: Number of attention heads (default 4).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm_scalar = nn.LayerNorm(dim)
        self.norm_image  = nn.LayerNorm(dim)
        self.attn        = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self,
                scalar_token: torch.Tensor,
                image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalar_token:  [B, 1, dim]
            image_tokens:  [B, T, dim]

        Returns:
            fused_scalar:  [B, 1, dim]
        """
        q = self.norm_scalar(scalar_token)   # Pre-LN on query
        k = self.norm_image(image_tokens)    # Pre-LN on key/value
        attn_out, _ = self.attn(query=q, key=k, value=k)
        return scalar_token + attn_out       # Residual: preserve spatial coords

class RIRDiffusionModel(torch.nn.Module):
    """
    Diffusion model for RIR (Room Impulse Response) generation.
    """
    
    def __init__(self,
                 device,
                 sample_size=None,
                 n_timesteps=1000,
                 # UNet architecture parameters
                 block_out_channels: Tuple[int, ...] = (32, 64, 128),
                 layers_per_block: Union[int, List[int]] = 2,
                 use_cross_attention: Optional[List[bool]] = None,
                 attention_head_dim: Union[int, List[int]] = 2,
                 norm_num_groups: Optional[int] = None,
                 use_mid_block: bool = True,
                 # Conditioning parameters
                 use_cond_encoder: bool = False,
                 encoder_hidden_dims: Optional[List[int]] = [64, 128],
                 input_cond_dim: int = 10,
                 # Image conditioning parameters
                 image_encoder: Optional['ImageEncoder'] = None,
                 fusion_enable: bool = True,
                 image_fusion: str = 'cross_attn',
                 # Guidance parameters (CFG - Classifier-Free Guidance)
                 guidance_enabled: bool = False,
                 guidance_dropout_prob: float = 0.1,
                 # Model I/O parameters
                 in_channels: int = 2,
                 out_channels: int = 2):
        """
        Args:
            device:        Device to run the model on
            sample_size:   Size of the input sample
            n_timesteps:   Number of diffusion timesteps

            UNet parameters:
            block_out_channels:  Tuple of output channels for each block level
            layers_per_block:    Number of ResNet layers per block (single int or list)
            use_cross_attention: List of booleans for each block level, None for auto
            attention_head_dim:  Dimension of attention heads
            norm_num_groups:     Number of groups for GroupNorm, None for auto
            use_mid_block:       Whether to use a middle block

            Conditioning parameters:
            use_cond_encoder:    Whether to use a conditioning encoder MLP
            encoder_hidden_dims: Hidden dimensions for the conditioning MLP layers
            input_cond_dim:      Input dimension of the scalar conditioning vector
                                 (9 for SoundSpaces without RT60, 10 for GTU with RT60)

            Image conditioning parameters:
            image_encoder:  Pre-built ImageEncoder instance, or None to disable image
                            conditioning.  The encoder's out_dim must equal the scalar
                            encoder's output dim (cross_attention_dim) so both token
                            types share the same cross-attention sequence.
            image_fusion:   How to combine the scalar token with image tokens before
                            the UNet.  'concat' concatenates them directly;
                            'cross_attn' (default) inserts a FusionBlock where the scalar token
                            attends to image tokens first (useful for ablation studies).
            fusion_enable:  When False and image_fusion='cross_attn', skip the FusionBlock
                            and concatenate the scalar token with image tokens directly.
                            Default True preserves the standard behaviour.

            Guidance parameters (CFG - Classifier-Free Guidance):
            guidance_enabled:      Whether to enable guidance dropout during training
            guidance_dropout_prob: Probability of dropping conditioning during training

            I/O parameters:
            in_channels:  Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()

        # -------- Config --------
        self.device = device
        self.n_timesteps = n_timesteps
        self.use_cond_encoder = use_cond_encoder
        self.sample_size = sample_size
        self.input_cond_dim = input_cond_dim
        self.guidance_enabled = guidance_enabled
        self.guidance_dropout_prob = guidance_dropout_prob
        self.image_fusion = image_fusion
        
        # -------- Auto-configure parameters --------
        norm_num_groups, use_cross_attention, down_block_types, up_block_types, mid_block_type = self._auto_configure_model_params(
            block_out_channels, layers_per_block, use_cross_attention, norm_num_groups, use_mid_block)
        
        # -------- Conditioning encoder --------
        if use_cond_encoder:
            # Build encoder layers
            encoder_layers = []
            current_dim = mlp_input_dim
            for hidden_dim in encoder_hidden_dims:
                encoder_layers.extend([
                    torch.nn.Linear(current_dim, hidden_dim),
                    torch.nn.ReLU(),
                ])
                current_dim = hidden_dim
            
            self.condition_encoder = torch.nn.Sequential(*encoder_layers).to(self.device)
            cross_attention_dim = encoder_hidden_dims[-1]
        else:
            self.condition_encoder = torch.nn.Identity()
            cross_attention_dim = input_cond_dim
        
        # Store cross_attention_dim for null conditioning
        self._cross_attention_dim = cross_attention_dim

        # -------- Image encoder + optional fusion block --------
        self.image_encoder = None
        self.fusion_block   = None
        if image_encoder is not None:
            assert image_encoder.out_dim == cross_attention_dim, f"image_encoder.out_dim ({image_encoder.out_dim}) must equal cross_attention_dim ({cross_attention_dim})"
            self.image_encoder = image_encoder.to(self.device)
            if image_fusion == 'cross_attn' and fusion_enable:
                self.fusion_block = FusionBlock(dim=cross_attention_dim).to(self.device)

        # -------- Base UNet Model --------
        # Only use cross_attention_dim if we have cross-attention blocks
        if not(any(use_cross_attention) or (use_mid_block and mid_block_type == "UNetMidBlock2DCrossAttn")):
            cross_attention_dim = None

        self.model = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            block_out_channels=block_out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            mid_block_type=mid_block_type,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups
        ).to(self.device)
        
        # Store configuration for reference
        self.config = {
            'sample_size': sample_size,
            'n_timesteps': n_timesteps,
            'block_out_channels': block_out_channels,
            'layers_per_block': layers_per_block,
            'use_cross_attention': use_cross_attention,
            'attention_head_dim': attention_head_dim,
            'norm_num_groups': norm_num_groups,
            'use_mid_block': use_mid_block,
            'use_cond_encoder': use_cond_encoder,
            'encoder_hidden_dims': encoder_hidden_dims if use_cond_encoder else None,
            'input_cond_dim': input_cond_dim,
            'use_image_conditioning': image_encoder is not None,
            'fusion_enable': fusion_enable,
            'image_fusion': image_fusion if image_encoder is not None else None,
            'guidance_enabled': guidance_enabled,
            'guidance_dropout_prob': guidance_dropout_prob,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'cross_attention_dim': cross_attention_dim,
            'down_block_types': down_block_types,
            'up_block_types': up_block_types,
            'mid_block_type': mid_block_type
        }
        
    
    # All model_config keys that map directly to __init__ parameters; derived keys are excluded.
    _CONSTRUCTOR_KEYS = {
        'sample_size', 'n_timesteps',
        'block_out_channels', 'layers_per_block', 'use_cross_attention',
        'attention_head_dim', 'norm_num_groups', 'use_mid_block', 'use_cond_encoder',
        'encoder_hidden_dims', 'input_cond_dim', 'image_fusion', 'fusion_enable', 'guidance_enabled',
        'guidance_dropout_prob', 'in_channels', 'out_channels'
    }

    @classmethod
    def init_from_config(cls, model_config, device, image_encoder=None):
        """Create an instance from a model_config dict, removing non-__init__ keys."""
        params = {k: v for k, v in model_config.items() if k in cls._CONSTRUCTOR_KEYS}
        return cls(device=device, image_encoder=image_encoder, **params)

    def _find_optimal_norm_groups(self, channels: Tuple[int, ...]) -> int:
        """Find the optimal number of groups for GroupNorm that divides all channel dimensions."""
        import math
        
        # Find GCD of all channel dimensions
        result = channels[0]
        for ch in channels[1:]:
            result = math.gcd(result, ch)
        
        # Common group sizes in order of preference
        preferred_groups = [32, 16, 8, 4, 2, 1]
        
        for groups in preferred_groups:
            if result >= groups and result % groups == 0:
                # Check if this divides all channels
                if all(ch % groups == 0 for ch in channels):
                    return groups
        
        return 1  # Fallback to 1 (equivalent to LayerNorm)
    
    def _auto_configure_model_params(self, block_out_channels, layers_per_block, use_cross_attention, norm_num_groups, use_mid_block):

        num_blocks = len(block_out_channels)
        # If layers_per_block is list, must match number of blocks
        assert isinstance(layers_per_block, int) or len(layers_per_block) == num_blocks, f"layers_per_block length must match block_out_channels length"
        
        # Auto-set norm_num_groups if not specified
        if norm_num_groups is None:
            # Find GCD of all channel dimensions for optimal group norm
            norm_num_groups = self._find_optimal_norm_groups(block_out_channels)
        
        # Auto-configure cross-attention locations if not specified
        if use_cross_attention is None:
            # Default: use cross-attention only at first block
            use_cross_attention = [False] * num_blocks
            use_cross_attention[0] = True  # First block
        
        # Ensure use_cross_attention matches number of blocks
        assert len(use_cross_attention) == num_blocks, \
            f"use_cross_attention length ({len(use_cross_attention)}) must match block_out_channels length ({num_blocks})"
        
        # Generate block types based on cross-attention configuration
        down_block_types = []
        up_block_types = []
        
        for i, use_ca in enumerate(use_cross_attention):
            if use_ca:
                down_block_types.append("CrossAttnDownBlock2D")
                up_block_types.insert(0, "CrossAttnUpBlock2D")
            else:
                down_block_types.append("DownBlock2D")
                up_block_types.insert(0, "UpBlock2D")
        
        # Configure middle block
        if use_mid_block:
            # Use cross-attention in middle block if any block uses it
            if any(use_cross_attention):
                mid_block_type = "UNetMidBlock2DCrossAttn"
            else:
                mid_block_type = "UNetMidBlock2D"
        else:
            mid_block_type = None

        return norm_num_groups, use_cross_attention, down_block_types, up_block_types, mid_block_type
    
    def get_model_params(self):
        """Get all trainable parameters (excludes frozen backbone weights)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_null_conditioning(self, batch_size: int, seq_len: int = 1) -> torch.Tensor:
        """
        Returns a zero conditioning tensor of shape [B, seq_len, cross_attention_dim].

        seq_len must match the conditioning sequence used during the forward pass
        so that CFG unconditional and conditional predictions have the same shape.
        """
        return torch.zeros(batch_size, seq_len, self._cross_attention_dim, device=self.device)
    
    def encode_conditioning(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Encode raw conditioning through the encoder.
        
        Args:
            cond: Raw conditioning tensor of shape [B, input_cond_dim]
            
        Returns:
            Encoded conditioning tensor of shape [B, 1, cross_attention_dim]
        """
        if self.use_cond_encoder:
            return self.condition_encoder(cond).unsqueeze(1)
        else:
            return cond.unsqueeze(1)
    
    def apply_guidance_dropout(self, cond_encoded: torch.Tensor) -> torch.Tensor:
        """
        Apply guidance dropout to the full conditioning sequence.
        Randomly zeros the entire sequence for a fraction of samples.

        The mask shape [B, 1, 1] broadcasts across all sequence positions and
        feature dimensions, so all tokens (scalar + image) are dropped together.

        Args:
            cond_encoded: [B, seq_len, cross_attention_dim]

        Returns:
            Conditioning with random dropout applied, same shape as input
        """
        batch_size, seq_len, _ = cond_encoded.shape
        dropout_mask = torch.rand(batch_size, 1, 1, device=self.device) < self.guidance_dropout_prob
        null_cond    = self.get_null_conditioning(batch_size, seq_len)
        return torch.where(dropout_mask, null_cond, cond_encoded)
    
    def forward(self, x, t, cond, images=None):
        """
        Forward pass through the model.

        Args:
            x:      Noisy input spectrogram [B, C, F, T]
            t:      Timestep tensor [B]
            cond:   Scalar conditioning vector [B, input_cond_dim]
            images: Image batch [B, N_img, 3, H, W]. Required when the model
                    was built with an image_encoder; must be None otherwise.
        """
        if cond.dim() == 3:
            cond = cond.squeeze(1)

        scalar_token = self.encode_conditioning(cond)       # [B, 1, D]

        if self.image_encoder is not None:
            image_tokens = self.image_encoder(images)       # [B, T, D]
            if self.fusion_block is not None:
                scalar_token = self.fusion_block(scalar_token, image_tokens)
            cond_encoded = torch.cat([scalar_token, image_tokens], dim=1)  # [B, 1+T, D]
        else:
            cond_encoded = scalar_token                     # [B, 1, D]

        if self.guidance_enabled and self.training:
            cond_encoded = self.apply_guidance_dropout(cond_encoded)

        return self.model(x, t, cond_encoded)
    
    @torch.no_grad()
    def generate(self, cond: torch.Tensor, shape: torch.Size, images=None,
                 num_inference_steps: int = None, guidance_scale: float = 1.0,
                 use_ddim: bool = False, verbose: bool = True):
        """
        Generate a synthetic RIR conditioned on input parameters with optional guidance.

        Args:
            cond:                Tensor [input_cond_dim] or [B, input_cond_dim]
            shape:               Shape of the output tensor
            images:              Image batch [B, N_img, 3, H, W], or None.
                                 Required when the model has an image_encoder.
            num_inference_steps: Reverse diffusion steps (< n_timesteps forces DDIM)
            guidance_scale:      CFG scale. 1.0 = no guidance; >1.0 = stronger effect
            use_ddim:            Use DDIM (faster) instead of DDPM
            verbose:             Show progress bar

        Returns:
            Generated RIR signal (CPU tensor)
        """
        self.model.eval()
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = cond.float().to(self.device)
        batch_size = cond.shape[0]

        # Build conditional encoding
        scalar_token = self.encode_conditioning(cond)       # [B, 1, D]
        if self.image_encoder is not None:
            image_tokens = self.image_encoder(images)       # [B, T, D]
            if self.fusion_block is not None:
                scalar_token = self.fusion_block(scalar_token, image_tokens)
            cond_encoded = torch.cat([scalar_token, image_tokens], dim=1)
        else:
            cond_encoded = scalar_token

        seq_len   = cond_encoded.shape[1]
        null_cond = self.get_null_conditioning(batch_size, seq_len)

        # Initialize with Gaussian noise
        noisy_rir = torch.randn(shape, device=self.device)

        # Set num_steps and DDIM 
        if num_inference_steps is None:
            num_inference_steps = self.n_timesteps
        elif num_inference_steps < self.n_timesteps:
            use_ddim = True  # Force DDIM for fewer steps

        # Set up inference scheduler
        if use_ddim:
            inference_scheduler = DDIMScheduler(num_train_timesteps=self.n_timesteps)
        else:
            inference_scheduler = DDPMScheduler(num_train_timesteps=self.n_timesteps)
        inference_scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(inference_scheduler.timesteps, desc="Generating", disable=not verbose):
            if guidance_scale != 1.0:
                # Unconditional prediction
                uncond_output = self.model(noisy_rir, t, null_cond)["sample"]
                # Conditional prediction
                cond_output = self.model(noisy_rir, t, cond_encoded)["sample"]
                # Guidance combination: uncond + scale * (cond - uncond)
                model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
            else:
                # Standard conditional generation (no guidance overhead)
                model_output = self.model(noisy_rir, t, cond_encoded)["sample"]

            noisy_rir = inference_scheduler.step(model_output, t, noisy_rir).prev_sample

        return noisy_rir.cpu()
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_config(self):
        """Print the model configuration in a readable format."""
        print("="*50)
        print("RIRDiffusionModel Configuration:")
        print("="*50)
        for key, value in self.config.items():
            print(f"{key:25s}: {value}")
        print(f"{'Total parameters':25s}: {self.count_parameters():,}")
        print("="*50)
        