import os
import time
import datetime
import logging
import torch
import torch.nn.functional as F
import shutil
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from utils.misc import save_metric, get_timestamped_logdir

from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.models.unets import UNet2DConditionModel
from typing import List, Tuple, Optional, Union

# ToDo:
#     - set torch.nn.Module as superclass ?
#     - use UNet2DConditionModel with the block_out_channels=(32, 64, 128, 128) (duplicate first/last block from up_block_types and down_block_types)
#     - try transformer based model
#     - try different scheduler
#     - understand whether a conditioning encoder is necessary
#     - use RT60(freq)

class RIRDiffusionModel(torch.nn.Module):
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
                 # Model I/O parameters
                 in_channels: int = 2,
                 out_channels: int = 2):
        """        
        Args:
            device: Device to run the model on
            sample_size: Size of the input sample
            n_timesteps: Number of diffusion timesteps
            
            UNet parameters:
            block_out_channels: Tuple of output channels for each block level
            layers_per_block: Number of ResNet layers per block (single int or list)
            use_cross_attention: List of booleans for each block level, None for auto
            attention_head_dim: Dimension of attention heads
            norm_num_groups: Number of groups for GroupNorm, None for auto
            use_mid_block: Whether to use a middle block
            
            Conditioning parameters:
            use_cond_encoder: Whether to use a conditioning encoder
            encoder_hidden_dims: Hidden dimensions for encoder layers
            input_cond_dim: Input dimension for conditioning
            
            I/O parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        # -------- Config --------
        self.device = device
        self.n_timesteps = n_timesteps
        self.use_cond_encoder = use_cond_encoder
        self.sample_size = sample_size
        self.input_cond_dim = input_cond_dim
        
        # -------- Auto-configure parameters --------
        norm_num_groups , use_cross_attention, down_block_types, up_block_types , mid_block_type = self._auto_configure_model_params(
            block_out_channels, layers_per_block, use_cross_attention, norm_num_groups, use_mid_block )
        
        # -------- Conditioning encoder --------
        if use_cond_encoder:
            # Build encoder layers
            encoder_layers = []
            current_dim = input_cond_dim
            for hidden_dim in encoder_hidden_dims:
                encoder_layers.extend([
                    torch.nn.Linear(current_dim, hidden_dim),
                    torch.nn.ReLU(),
                ])
                current_dim = hidden_dim
            
            self.condition_encoder = torch.nn.Sequential(*encoder_layers).to(self.device)
            cross_attention_dim = encoder_hidden_dims[-1]  # Last hidden dimension for cross-attention
        else:
            self.condition_encoder = torch.nn.Identity()
            cross_attention_dim = input_cond_dim
        
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
            'in_channels': in_channels,
            'out_channels': out_channels,
            'cross_attention_dim': cross_attention_dim,
            'down_block_types': down_block_types,
            'up_block_types': up_block_types,
            'mid_block_type': mid_block_type
        }
        
    
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
        """Get all trainable parameters."""
        model_params = list(self.model.parameters()) + list(self.condition_encoder.parameters()) \
            if self.use_cond_encoder else self.model.parameters()
        return model_params
    
    def forward(self, x, t, cond):
        """Forward pass through the model."""
        if self.use_cond_encoder:
            cond = self.condition_encoder(cond)
        
        prediction = self.model(x, t, cond)
        return prediction
    
    @torch.no_grad()
    def generate(self, cond: torch.Tensor, shape: torch.Size, num_steps: int = 50):
        """
        Generate a synthetic RIR conditioned on input parameters.
        
        Args:
            cond: Tensor of shape [input_cond_dim] or [B, input_cond_dim] containing conditioning
            shape: Shape of the output tensor
            num_steps: Number of reverse diffusion steps
            
        Returns:
            Generated RIR signal
        """
        self.model.eval()
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = cond.float().to(self.device)
        
        # Use encoder if defined
        if self.use_cond_encoder:
            cond = self.condition_encoder(cond)
        cond = cond.unsqueeze(1)  # [B, 1, C] for model
        
        # Initialize with Gaussian noise
        noisy_rir = torch.randn(shape, device=self.device)
        
        # Set up inference timesteps
        inference_scheduler = DDPMScheduler(num_train_timesteps=self.n_timesteps)
        inference_scheduler.set_timesteps(num_steps)
        
        for t in inference_scheduler.timesteps:
            model_output = self.model(noisy_rir, t, cond)["sample"]
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

