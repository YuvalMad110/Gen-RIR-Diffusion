"""
Data loading utilities for RIR inference.

Functions list:
- load_model_and_config: Load trained diffusion model and its configuration.
- load_dataset_conditions: Load conditions and RIRs from dataset.

"""

import os
import json
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

from utils.signal_proc import  waveform_to_spectrogram, spectrogram_to_waveform, calculate_edc, estimate_decay_k_factor

def load_model_and_data_info(model_path: str, device: torch.device, model_class) -> Tuple[Any, Dict]:
    """Load trained diffusion model and data_info from checkpoint."""
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Get model config and data info directly from checkpoint
    model_config = checkpoint['model_config']
    data_info = checkpoint['data_info']

    # Extract n_timesteps from model_config
    n_timesteps = model_config['n_timesteps']

    # Create model from config, excluding derived fields
    model_params = {k: v for k, v in model_config.items()
                    if k not in ("n_timesteps", "cross_attention_dim", "down_block_types",
                                "up_block_types", "mid_block_type")}

    model = model_class(device=device, n_timesteps=n_timesteps, **model_params)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"Model loaded on {device}")
    print(f"  Sample size: {model_config['sample_size']}, Timesteps: {n_timesteps}")
    if hasattr(model, 'guidance_enabled'):
        print(f"  Guidance: enabled={model.guidance_enabled}, dropout={model.guidance_dropout_prob}")

    return model, data_info


def load_dataset_conditions(dataset, n_samples: int, data_info: Dict,
                            rir_indices: Optional[List[int]] = None
                            ) -> Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray], List[int], Optional[torch.Tensor]]:
    """Load conditions and RIRs from dataset."""
    # Select indices
    if rir_indices is None:
        rir_indices = np.random.choice(len(dataset), size=n_samples, replace=False).tolist()
    else:
        if len(rir_indices) != n_samples:
            print(f"Warning: Using first {n_samples} of {len(rir_indices)} provided indices")
            rir_indices = rir_indices[:n_samples]
    
    print(f"Loading dataset samples at indices: {rir_indices}")
    
    # Get STFT params - must exist in data_info
    hop_length = data_info['hop_length']
    n_fft = data_info['n_fft']
    
    conditions = []
    real_rirs_wave = []
    real_rirs_spec = []
    
    for idx in rir_indices:
        rir, room_dim, mic_loc, speaker_loc, rt60 = dataset[idx]
        
        # Convert tensors to numpy
        rir = _to_numpy(rir)
        room_dim = _to_numpy(room_dim)
        mic_loc = _to_numpy(mic_loc)
        speaker_loc = _to_numpy(speaker_loc)
        rt60 = _to_numpy(rt60)
        
        # Handle RIR format
        if rir.ndim == 3 and rir.shape[0] == 2:
            rir_spec = rir
            rir_wave = spectrogram_to_waveform(rir, hop_length, n_fft)
        else:
            rir_wave = rir.squeeze()
            rir_spec = waveform_to_spectrogram(rir_wave, hop_length, n_fft)
        
        real_rirs_wave.append(rir_wave)
        real_rirs_spec.append(rir_spec)
        condition = np.concatenate([room_dim, mic_loc, speaker_loc, [rt60]])
        conditions.append(condition)
    
    conditions_tensor = torch.tensor(conditions, dtype=torch.float32)
    
    # Calculate k-factors if scaling was used in training
    k_factors = None
    if data_info.get('scale_rir', False):
        real_tensor = torch.stack([torch.tensor(r, dtype=torch.float32) for r in real_rirs_wave])
        edc = calculate_edc(real_tensor)
        k_factors, _ = estimate_decay_k_factor(edc, data_info['sr_target'], data_info['db_cutoff'])
        print(f"Calculated k-factors: {k_factors.tolist()}")
    
    return conditions_tensor, real_rirs_wave, real_rirs_spec, rir_indices, k_factors


def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.asarray(x)


