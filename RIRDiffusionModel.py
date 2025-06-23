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
from diffusers.models.unets import UNet2DConditionModel

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
                 use_cond_encoder=False,
                 light_mode=False):
        """
        Initialize the RIR generator.
        """
        super().__init__()
        # -------- Cfg --------
        self.device = device
        self.n_timesteps = n_timesteps
        self.use_cond_encoder = use_cond_encoder
        self.light_mode = light_mode
        self.sample_size = sample_size
        # -------- Conditioning encoder --------
        if use_cond_encoder:
            cross_attention_dim = 128 # length of the context embedding
            self.condition_encoder = torch.nn.Sequential(
                torch.nn.Linear(10, 64),
                torch.nn.ReLU(),
                torch.nn.Linear (64, cross_attention_dim),
                torch.nn.ReLU(),
            ).to(self.device)
        else:
            self.condition_encoder = torch.nn.Identity()
            cross_attention_dim = 10

        # -------- Base Model --------
        if not light_mode:
            # Operative model
            self.model = UNet2DConditionModel(
                sample_size=sample_size,
                in_channels=2,
                out_channels=2,
                layers_per_block=2,
                cross_attention_dim=cross_attention_dim,
                block_out_channels=(32, 64, 128),
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D", "CrossAttnUpBlock2D"),
                mid_block_type="UNetMidBlock2DCrossAttn",
                attention_head_dim=2,
                norm_num_groups=8
            ).to(self.device)
        else:
            # light mode: for debug only (faster training)
            self.model = UNet2DConditionModel(
                sample_size=sample_size,
                in_channels=2,
                out_channels=2,
                layers_per_block=1,
                cross_attention_dim=cross_attention_dim,
                block_out_channels=(16, 32),  # ultra-thin
                down_block_types=("DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D"),
                mid_block_type=None,
                attention_head_dim=1,
                norm_num_groups=4
            ).to(self.device)


    def get_model_params(self):
        model_params = list(self.model.parameters()) + list(self.condition_encoder.parameters()) \
            if self.use_cond_encoder else self.model.parameters()
        return model_params

    
    def forward(self, x, t, cond):

        if self.use_cond_encoder:
            cond = self.condition_encoder(cond)

        prediction = self.model(x, t, cond)
        return prediction


    @torch.no_grad()
    def generate(self, cond: torch.Tensor, shape: torch.Size, num_steps: int = 50):
        """
        Generate a synthetic RIR conditioned on input parameters.

        Args:
            cond (torch.Tensor): Tensor of shape [10] or [B, 10] containing the conditioning parameters.
            length (int): The desired RIR signal length.
            num_steps (int): Number of reverse diffusion steps (can be less than n_timesteps for faster sampling).

        Returns:
            torch.Tensor: Generated RIR signal of shape [1, 1, length]
        """
        self.model.eval()
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = cond.float().to(self.device)  # [1, 10]

        # Use encoder if defined
        if self.use_cond_encoder:
            cond = self.condition_encoder(cond)  # [1, 128]
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

