#!/usr/bin/env python3
"""
RIR Diffusion Model Inference Script - Two Modes Version

This script loads a trained RIR diffusion model and generates RIRs with two modes:
Mode 1: Random conditions (like before)
Mode 2: Conditions from dataset with comparison to real RIRs

Usage:
    python run_inference.py --model_path /path/to/model_best.pth.tar --nRIR 5 --mode 1
    python run_inference.py --model_path /path/to/model_best.pth.tar --nRIR 5 --mode 2 --dataset_path /path/to/dataset

Author: Yuval (Two Modes)
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import json
from typing import Optional, Tuple, List, Dict, Any
import soundfile as sf
from tqdm import tqdm
from diffusers import DDPMScheduler
from RIRDiffusionModel import RIRDiffusionModel
from data.rir_dataset import load_rir_dataset
from utils.signal_edc import create_edc_plots_mode2
from utils.signal_proc import calculate_edc, estimate_decay_k_factor, undo_rir_scaling, apply_rir_scaling
import glob
import random
from scipy.signal import convolve


"""
finished runs:
scaled - 128S 100E 1000T
    /home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/Jul02_18-11-05_dsief06

raw -  128S 300E 1000T
    /home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/Jun24_00-37-28_dsief07/model_best.pth.ta
"""
def load_model_and_config(model_path: str, device: torch.device) -> Tuple[RIRDiffusionModel, dict]:
    """
    Load the trained diffusion model and its configuration.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config_dict)
    """
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration from checkpoint
    config = {
        'sample_size': checkpoint.get('sample_size', (64, 64)),
        'n_timesteps': checkpoint.get('n_timesteps', 1000),
        'use_cond_encoder': checkpoint.get('use_cond_encoder', False),
        'light_mode': checkpoint.get('light_mode', False),
        'data_info': checkpoint.get('data_info', {})
    }
    
    # Initialize model with config
    model = RIRDiffusionModel(
        device=device,
        sample_size=config['sample_size'],
        n_timesteps=config['n_timesteps'],
        use_cond_encoder=config['use_cond_encoder'],
        light_mode=config['light_mode']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Sample size: {config['sample_size']}")
    print(f"Training timesteps: {config['n_timesteps']}")
    print(f"Using conditional encoder: {config['use_cond_encoder']}")
    
    return model, config

def generate_random_conditions(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Generate realistic RIR conditioning parameters for a batch.
    
    Args:
        batch_size: Number of conditions to generate
        device: Device to create tensors on
    
    Returns:
        Tensor of shape [batch_size, 10] containing conditioning parameters
        Order: [room_length, room_width, room_height, mic_x, mic_y, mic_z, 
                speaker_x, speaker_y, speaker_z, rt60]
    """
    conditions = []
    
    for _ in range(batch_size):
        # Room dimensions [length, width, height] - 3 to 8 meters
        room_dim = torch.rand(3) * 5 + 3  # 3-8 meters
        
        # Microphone location - at least 0.5m from walls
        mic_x = torch.rand(1) * (room_dim[0] - 1.0) + 0.5
        mic_y = torch.rand(1) * (room_dim[1] - 1.0) + 0.5
        mic_z = torch.rand(1) * (room_dim[2] - 1.0) + 0.5
        mic_loc = torch.cat([mic_x, mic_y, mic_z])
        
        # Speaker location - at least 0.5m from walls and 1m from microphone
        valid_speaker = False
        attempts = 0
        while not valid_speaker and attempts < 50:
            speaker_x = torch.rand(1) * (room_dim[0] - 1.0) + 0.5
            speaker_y = torch.rand(1) * (room_dim[1] - 1.0) + 0.5
            speaker_z = torch.rand(1) * (room_dim[2] - 1.0) + 0.5
            speaker_loc = torch.cat([speaker_x, speaker_y, speaker_z])
            
            # Check distance from microphone
            if torch.norm(speaker_loc - mic_loc) >= 1.0:
                valid_speaker = True
            attempts += 1
        
        if not valid_speaker:
            # Fallback positioning
            speaker_loc = torch.tensor([room_dim[0] - 0.5, room_dim[1] - 0.5, room_dim[2] - 0.5])
        
        # RT60 - 0.3 to 1.5 seconds
        rt60 = torch.rand(1) * 1.2 + 0.3
        
        # Combine all conditions
        condition = torch.cat([room_dim, mic_loc, speaker_loc, rt60])
        conditions.append(condition)
    
    return torch.stack(conditions, dim=0).to(device).float()

def load_dataset_conditions(dataset, nRIR: int, data_info: dict, use_spectrogram: bool, 
                           rir_indices: Optional[List[int]] = None) -> Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Load conditions and RIRs from dataset.
    
    Args:
        dataset: Loaded RIR dataset
        nRIR: Number of RIRs to load
        data_info: Data configuration info
        use_spectrogram: Whether to return spectrograms or waveforms
        rir_indices: Optional list of specific indices to load
    """
    if rir_indices is None:
        # Randomly select indices
        rir_indices = np.random.choice(len(dataset), size=nRIR, replace=False).tolist()
    else:
        # Use provided indices
        if len(rir_indices) != nRIR:
            print(f"Warning: Provided {len(rir_indices)} indices but requested {nRIR} RIRs. Using first {nRIR} indices.")
            rir_indices = rir_indices[:nRIR]
    
    conditions = []
    real_rirs_waveform = []
    real_rirs_spectrogram = []
    
    print(f"Loading dataset samples at indices: {rir_indices}")
    
    # Get STFT parameters
    hop_length = data_info.get('hop_length', 64)
    n_fft = data_info.get('n_fft', 128)
    
    for idx in rir_indices:
        # Load from dataset: rir, room_dim, mic_loc, speaker_loc, rt60
        rir, room_dim, mic_loc, speaker_loc, rt60 = dataset[idx]
        
        # Convert to numpy if needed
        if torch.is_tensor(rir):
            rir = rir.cpu().numpy()
        if torch.is_tensor(room_dim):
            room_dim = room_dim.cpu().numpy()
        if torch.is_tensor(mic_loc):
            mic_loc = mic_loc.cpu().numpy()
        if torch.is_tensor(speaker_loc):
            speaker_loc = speaker_loc.cpu().numpy()
        if torch.is_tensor(rt60):
            rt60 = rt60.cpu().numpy()
        
        # Handle RIR format based on what dataset returns
        if rir.ndim == 3 and rir.shape[0] == 2:
            # Dataset returned spectrogram [2, F, T]
            rir_spec = rir
            rir_wave = spectrogram_to_waveform(rir, hop_length, n_fft)
        else:
            # Dataset returned waveform
            rir_wave = rir.squeeze()
            rir_spec = waveform_to_spectrogram(rir_wave, hop_length, n_fft)
        
        # Store both formats
        real_rirs_waveform.append(rir_wave)
        real_rirs_spectrogram.append(rir_spec)
        
        # Combine conditions in the expected order
        condition = np.concatenate([room_dim, mic_loc, speaker_loc, [rt60]])
        conditions.append(condition)
    
    # Convert to tensor
    conditions_tensor = torch.tensor(conditions, dtype=torch.float32)

    # Check if scaling was used in training
    scale_rir_enabled = data_info.get('scale_rir', False)
    k_factors = None
    if scale_rir_enabled:
        # Calculate k factors for real RIRs to enable unscaling of generated RIRs
        real_rirs_tensor = torch.stack([torch.tensor(rir, dtype=torch.float32) for rir in real_rirs_waveform])
        edc = calculate_edc(real_rirs_tensor)
        sr_target = data_info.get('sr_target', 22050)
        db_cutoff = data_info.get('db_cutoff', -40.0)
        k_factors = estimate_decay_k_factor(edc, sr_target, db_cutoff)
        print(f"Calculated k-factors for unscaling: {k_factors}")
    
    return conditions_tensor, real_rirs_waveform, real_rirs_spectrogram, rir_indices, k_factors

def prepare_scaled_unscaled_datasets(real_rirs_wave: List[np.ndarray], generated_rirs_wave: List[np.ndarray], 
                                   k_factors: torch.Tensor, sr: int, n_fft: int, hop_length: int):
    """
    1. Scale real rir
    2. Unscale generated rir 
    3. Generate spectrograms of the above
    """    
    # Convert to tensors for processing
    real_rirs_tensor = torch.stack([torch.tensor(rir, dtype=torch.float32) for rir in real_rirs_wave])
    generated_rirs_tensor = torch.stack([torch.tensor(rir, dtype=torch.float32) for rir in generated_rirs_wave])
    
    # === SCALE REAL RIR ===
    # Scale real RIRs using the same k-factors
    scaled_real_tensor = apply_rir_scaling(real_rirs_tensor, k_factors, sr)
    real_rirs_wave_scaled = [rir.cpu().numpy() for rir in scaled_real_tensor]
    
    # Convert scaled real RIRs to spectrograms
    real_rirs_spec_scaled = []
    for rir in real_rirs_wave_scaled:
        spec = waveform_to_spectrogram(rir, hop_length, n_fft)
        real_rirs_spec_scaled.append(spec)
    
    # === UNSCALED GENERATED RIR ===    
    # Unscale generated RIRs
    unscaled_generated_tensor = undo_rir_scaling(generated_rirs_tensor, k_factors, sr)
    gen_rirs_wave_unscaled = [rir.cpu().numpy() for rir in unscaled_generated_tensor]
    
    # Convert unscaled generated RIRs to spectrograms
    gen_rirs_spec_unscaled = []
    for rir in gen_rirs_wave_unscaled:
        spec = waveform_to_spectrogram(rir, hop_length, n_fft)
        gen_rirs_spec_unscaled.append(spec)
    
    return  gen_rirs_wave_unscaled, gen_rirs_spec_unscaled, real_rirs_wave_scaled, real_rirs_spec_scaled

def generate_rirs_batch(model: RIRDiffusionModel, conditions: torch.Tensor, device: torch.device, 
                       config: dict) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate RIRs using the diffusion model with given conditions.
    
    Args:
        model: Trained diffusion model
        conditions: Conditioning parameters [nRIR, 10]
        device: Device for computation
        config: Model configuration
        
    Returns:
        Tuple of (generated_rirs_spectrogram, generated_rirs_waveform)
    """
    # Get parameters from config
    sample_size = config['sample_size']
    n_timesteps = config['n_timesteps']
    data_info = config.get('data_info', {})
    nRIR = conditions.shape[0]
    
    print(f"Generating {nRIR} RIRs with {n_timesteps} timesteps...")
    
    # Move conditions to device
    conditions = conditions.to(device)
    
    # Setup scheduler - exactly as in training
    scheduler = DDPMScheduler(num_train_timesteps=n_timesteps)
    scheduler.set_timesteps(n_timesteps)  # Use all training timesteps for best quality
    
    print(f"Using conditions shape: {conditions.shape}")
    
    # Prepare conditioning for model
    with torch.no_grad():
        if model.use_cond_encoder:
            encoded_conditions = model.condition_encoder(conditions)
        else:
            encoded_conditions = conditions
        
        # Add sequence dimension if needed
        if encoded_conditions.dim() == 2:
            encoded_conditions = encoded_conditions.unsqueeze(1)
    
    print(f"Encoded conditions shape: {encoded_conditions.shape}")
    
    # Initialize with pure noise
    batch_size = nRIR
    channels = 2  # Real and imaginary parts
    samples = torch.randn(batch_size, channels, *sample_size, device=device)
    
    print(f"Initial samples shape: {samples.shape}")
    
    # Denoising loop
    model.eval()
    with torch.no_grad():
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
            # Prepare timestep
            timestep = t.expand(batch_size).to(device)
            
            # Model forward pass
            try:
                model_output = model(samples, timestep, encoded_conditions)
                
                # Extract noise prediction
                if isinstance(model_output, dict):
                    noise_pred = model_output.get("sample", model_output.get("pred", model_output))
                else:
                    noise_pred = model_output
                
                # Scheduler step
                samples = scheduler.step(noise_pred, t, samples).prev_sample
                
            except Exception as e:
                print(f"Error at timestep {i}: {e}")
                break
    
    # Convert to numpy
    generated_specs = samples.cpu().numpy()
    
    # Convert spectrograms to waveforms
    generated_rirs = []
    print("Converting spectrograms to waveforms...")
    
    for i in range(nRIR):
        spec = generated_specs[i]  # [2, freq, time]
        waveform = spectrogram_to_waveform(
            spec, 
            data_info.get('hop_length', 64),
            data_info.get('n_fft', 128)
        )
        generated_rirs.append(waveform)
    # convert to list
    generated_specs = [generated_specs[i] for i in range(nRIR)]
    
    return generated_specs, generated_rirs

def waveform_to_spectrogram(waveform: np.ndarray, hop_length: int = 64, n_fft: int = 128) -> np.ndarray:
    """
    Convert waveform to spectrogram using the same approach as the training dataset.
    
    Args:
        waveform: Waveform signal
        hop_length: Hop length for STFT
        n_fft: FFT size for STFT
        
    Returns:
        Spectrogram of shape [2, freq, time] (real, imag)
    """
    # Convert to torch tensor
    rir_tensor = torch.from_numpy(waveform).float()
    
    # Create Hann window (same as in training)
    window = torch.hann_window(n_fft, device=rir_tensor.device)
    
    # Apply STFT (same as training)
    rir_stft = torch.stft(
        rir_tensor.squeeze(), 
        n_fft=n_fft, 
        hop_length=hop_length, 
        return_complex=True, 
        window=window
    )
    
    # Stack real and imaginary parts (same as training)
    rir_spec = torch.stack((rir_stft.real, rir_stft.imag), dim=0)  # [2, F, T]
    
    return rir_spec.cpu().numpy()

def spectrogram_to_waveform(spectrogram: np.ndarray, hop_length: int = 64, n_fft: int = 128) -> np.ndarray:
    """
    Convert complex spectrogram back to waveform using the same approach as the dataset.
    
    Args:
        spectrogram: Complex spectrogram of shape [2, freq, time] (real, imag)
        hop_length: Hop length used in STFT
        n_fft: FFT size used in STFT
        
    Returns:
        Reconstructed waveform
    """
    # Convert numpy to torch tensor
    spec_tensor = torch.from_numpy(spectrogram)  # [2, F, T]
    
    # Reconstruct complex spectrogram from real and imaginary parts
    complex_spec = torch.complex(spec_tensor[0], spec_tensor[1])  # [F, T]
    
    # Create Hann window (same as in dataset)
    window = torch.hann_window(n_fft, device=complex_spec.device)
    
    # Use torch.istft (inverse of torch.stft used in dataset)
    try:
        waveform = torch.istft(
            complex_spec, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window,
            return_complex=False
        )
    except Exception as e:
        print(f"Warning: torch.istft failed ({e}), using librosa fallback")
        # Fallback to librosa method
        complex_spec_np = complex_spec.cpu().numpy()
        waveform = librosa.istft(complex_spec_np, hop_length=hop_length, n_fft=n_fft)
        waveform = torch.from_numpy(waveform)
    
    # Convert back to numpy and ensure it's 1D
    return waveform.cpu().numpy().squeeze()

def format_condition_string(condition: np.ndarray, mode: str = "room") -> str:
    """Format condition parameters for display."""
    if mode == "room":
        room_dims = condition[:3]
        rt60 = condition[-1]
        return f"Room: {room_dims[0]:.1f}×{room_dims[1]:.1f}×{room_dims[2]:.1f}m, RT60: {rt60:.2f}s"
    elif mode == "locations":
        mic_loc = condition[3:6]
        speaker_loc = condition[6:9]
        return f"Mic: ({mic_loc[0]:.1f},{mic_loc[1]:.1f},{mic_loc[2]:.1f}), Src: ({speaker_loc[0]:.1f},{speaker_loc[1]:.1f},{speaker_loc[2]:.1f})"

def create_base_plot_mode1(rirs: List[np.ndarray], conditions: np.ndarray, sr: int, save_path: str):
    """
    Create plot for Mode 1: Random conditions (2 columns: waveform + spectrogram).
    
    Args:
        rirs: List of generated RIR waveforms
        conditions: Conditioning parameters used
        sr: Sample rate
        save_path: Directory to save plot
    """
    n_rirs = len(rirs)
    fig, axes = plt.subplots(n_rirs, 2, figsize=(16, 4*n_rirs))
    
    if n_rirs == 1:
        axes = axes.reshape(1, -1)
    
    for i, rir in enumerate(rirs):
        # Left column: Waveform
        time = np.arange(len(rir)) / sr
        axes[i, 0].plot(time, rir, linewidth=0.8, color='blue')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Add condition info to title
        condition_str = format_condition_string(conditions[i], "room")
        axes[i, 0].set_title(f'Generated RIR #{i+1} - {condition_str}')
        
        if i == n_rirs - 1:
            axes[i, 0].set_xlabel('Time (s)')
        
        # Right column: Spectrogram
        D = librosa.stft(rir, hop_length=512, n_fft=2048)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        img = librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', 
                                      y_axis='hz', ax=axes[i, 1], cmap='viridis')
        
        # Add location info to spectrogram title
        location_str = format_condition_string(conditions[i], "locations")
        axes[i, 1].set_title(f'Generated RIR #{i+1} Spectrogram - {location_str}')
        
        if i == n_rirs - 1:
            axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('Frequency (Hz)')
        
        # Add colorbar for the last spectrogram
        if i == n_rirs - 1:
            fig.colorbar(img, ax=axes[:, 1], format='%+2.0f dB', shrink=0.6)
    
    plt.tight_layout()
    
    plot_path = Path(save_path) / "generated_rirs_mode1.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Mode 1 plot saved to: {plot_path}")

def create_base_plot_mode2(real_rirs_wave: List[np.ndarray], generated_rirs_wave: List[np.ndarray], 
                     real_rirs_spec: List[np.ndarray], generated_rirs_spec: List[np.ndarray],
                     conditions: np.ndarray, rir_indices: List[int], sr: int, save_path: str, title: str):
    """
    Create plot for Mode 2: Dataset comparison (4 columns: real waveform, generated waveform, real spectrogram, generated spectrogram).
    
    Args:
        real_rirs_wave: List of real RIR waveforms from dataset
        generated_rirs_wave: List of generated RIR waveforms
        real_rirs_spec: List of real RIR spectrograms from dataset
        generated_rirs_spec: List of generated RIR spectrograms
        conditions: Conditioning parameters used
        rir_indices: Dataset indices used
        sr: Sample rate
        save_path: Directory to save plot
    """
    n_rirs = len(real_rirs_wave)
    fig, axes = plt.subplots(n_rirs, 4, figsize=(20, 4*n_rirs))
    fig.suptitle(title)

    if n_rirs == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_rirs):
        real_rir_wave = real_rirs_wave[i]
        gen_rir_wave = generated_rirs_wave[i]
        real_rir_spec = real_rirs_spec[i]
        gen_rir_spec = generated_rirs_spec[i]
        idx = rir_indices[i]
        
        # Column 1: Real waveform
        time_real = np.arange(len(real_rir_wave)) / sr
        axes[i, 0].plot(time_real, real_rir_wave, linewidth=0.8, color='green')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        condition_str = format_condition_string(conditions[i], "room")
        axes[i, 0].set_title(f'Real RIR #{idx} - {condition_str}')
        
        if i == n_rirs - 1:
            axes[i, 0].set_xlabel('Time (s)')
        
        # Column 2: Generated waveform
        time_gen = np.arange(len(gen_rir_wave)) / sr
        axes[i, 1].plot(time_gen, gen_rir_wave, linewidth=0.8, color='blue')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title(f'Generated RIR #{idx}')
        
        if i == n_rirs - 1:
            axes[i, 1].set_xlabel('Time (s)')
        
        # Column 3: Real spectrogram (from dataset spectrogram)
        real_magnitude = np.sqrt(real_rir_spec[0]**2 + real_rir_spec[1]**2)
        real_db = 20 * np.log10(real_magnitude + 1e-8)
        
        im1 = axes[i, 2].imshow(real_db, aspect='auto', origin='lower', cmap='viridis')
        
        location_str = format_condition_string(conditions[i], "locations")
        axes[i, 2].set_title(f'Real RIR #{idx} Spectrogram ')
        axes[i, 2].set_ylabel('Frequency Bin')
        
        if i == n_rirs - 1:
            axes[i, 2].set_xlabel('Time Frame')
        
        # Column 4: Generated spectrogram (from model output)
        gen_magnitude = np.sqrt(gen_rir_spec[0]**2 + gen_rir_spec[1]**2)
        gen_db = 20 * np.log10(gen_magnitude + 1e-8)
        
        im2 = axes[i, 3].imshow(gen_db, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 3].set_title(f'Generated RIR #{idx} Spectrogram - {location_str}')
        axes[i, 3].set_ylabel('Frequency Bin')
        
        if i == n_rirs - 1:
            axes[i, 3].set_xlabel('Time Frame')

        
    plt.tight_layout()

    # # uncomment to show colorbar
    # # Adjust layout to make room for colorbars
    # plt.subplots_adjust(right=0.92)
    
    # # Add colorbars positioned to the right of each column
    # # Position colorbars for real spectrograms (column 3)
    # cbar_ax1 = fig.add_axes([0.93, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    # cbar1 = fig.colorbar(im1, cax=cbar_ax1, format='%+2.0f dB')
    # cbar1.set_label('Real RIR (dB)', rotation=270, labelpad=15)
    
    # # Position colorbars for generated spectrograms (column 4)
    # cbar_ax2 = fig.add_axes([0.96, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    # cbar2 = fig.colorbar(im2, cax=cbar_ax2, format='%+2.0f dB')
    # cbar2.set_label('Generated RIR (dB)', rotation=270, labelpad=15)
    if save_path.endswith(".png"):
        plot_path = save_path
    else:
        plot_path = Path(save_path) / "rirs_comparison_mode2.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Mode 2 comparison plot saved to: {plot_path}")

def save_audio_files(rirs: List[np.ndarray], save_path: str, sr: int = 22050, prefix: str = "generated"):
    """Save RIRs as audio files."""
    audio_dir = Path(save_path) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for i, rir in enumerate(rirs):
        filename = audio_dir / f"{prefix}_rir_{i+1:03d}.wav"
        sf.write(filename, rir, sr)
    
    print(f"{prefix.capitalize()} audio files saved to: {audio_dir}")

def save_generation_stats(generated_rirs: List[np.ndarray], conditions: np.ndarray, config: dict, 
                         model_path: str, save_path: str, sr: int, mode: int, 
                         real_rirs: Optional[List[np.ndarray]] = None, rir_indices: Optional[List[int]] = None):
    """Save generation statistics and parameters."""
    condition_names = [
        'room_length', 'room_width', 'room_height',
        'mic_x', 'mic_y', 'mic_z',
        'speaker_x', 'speaker_y', 'speaker_z',
        'rt60'
    ]
    
    # Analyze conditions
    conditions_analysis = {}
    for i, name in enumerate(condition_names):
        values = conditions[:, i]
        conditions_analysis[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values.tolist()
        }
    
    # Overall statistics
    stats = {
        "mode": mode,
        "n_generated": len(generated_rirs),
        "sample_rate": sr,
        "model_path": model_path,
        "generated_waveform_lengths": [len(rir) for rir in generated_rirs],
        "generated_max_amplitudes": [float(np.max(np.abs(rir))) for rir in generated_rirs],
        "model_config": {
            "sample_size": config['sample_size'],
            "n_timesteps": config['n_timesteps'],
            "use_cond_encoder": config['use_cond_encoder'],
            "light_mode": config['light_mode']
        },
        "data_info": config.get('data_info', {}),
        "conditions_analysis": conditions_analysis
    }
    
    # Add mode 2 specific stats
    if mode == 2 and real_rirs is not None:
        stats.update({
            "dataset_indices": rir_indices,
            "real_waveform_lengths": [len(rir) for rir in real_rirs],
            "real_max_amplitudes": [float(np.max(np.abs(rir))) for rir in real_rirs]
        })
    
    stats_path = Path(save_path) / f"generation_stats_mode{mode}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_path}")

# --------------------------- Speech convolution ---------------------------
def find_librispeech_files(speech_path: str, speech_id: Optional[List[str]] = None, n_speech_files: int = 3) -> List[str]:
    """Find LibriSpeech audio files efficiently."""
    speech_path = Path(speech_path)
    
    if speech_id is not None:
        # Case 1: Specific speech_ids provided - look for speech_path/speech_id.wav for each ID
        found_files = []
        for sid in speech_id:
            target_file = speech_path / f"{sid}.wav"
            if target_file.exists():
                found_files.append(str(target_file))
                print(f"Found file: {target_file}")
            else:
                print(f"File not found: {target_file}")
        
        if not found_files:
            print(f"No files found for speech_ids: {speech_id}")
            return []
        
        print(f"Found {len(found_files)} files for {len(speech_id)} speech_ids")
        return found_files
    
    else:
        # Case 2: No speech_id - randomly sample n_speech_files efficiently
        print(f"Searching for {n_speech_files} random audio files...")
        
        found_files = []
        
        # Use iterative sampling to avoid loading entire directory
        for root, dirs, files in os.walk(speech_path):
            for file in files:
                if file.lower().endswith(('.wav', '.flac')):
                    found_files.append(os.path.join(root, file))
                    
                    # stop after collecting 10x the requested rir (for randomness)
                    if len(found_files) >= n_speech_files * 10:
                        break
            
            if len(found_files) >= n_speech_files * 10:
                break
        
        if not found_files:
            print(f"No audio files found in {speech_path}")
            return []
        
        # Randomly sample from collected files
        n_files_to_use = min(n_speech_files, len(found_files))
        selected_files = random.sample(found_files, n_files_to_use)
        
        print(f"Found {len(found_files)} audio files, selected {len(selected_files)}")
        return selected_files

def load_speech_file(file_path: str, target_sr: int = 22050, max_duration: float = 10.0) -> Tuple[np.ndarray, int]:
    """Load and preprocess speech file."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Limit duration if too long
        max_samples = int(max_duration * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
            
        return audio, target_sr
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def convolve_speech_with_rir(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve speech signal with RIR using scipy.signal.convolve."""
    speech = speech.squeeze()
    rir = rir.squeeze()
    
    # Normalize RIR to prevent amplification
    if np.max(np.abs(rir)) > 0:
        rir = rir / np.max(np.abs(rir))
    
    # Convolve speech with RIR
    reverb_speech = convolve(speech, rir, mode='full')
    
    # Normalize output to prevent clipping
    if np.max(np.abs(reverb_speech)) > 0:
        reverb_speech = reverb_speech / np.max(np.abs(reverb_speech)) * 0.95
    
    return reverb_speech

def process_speech_convolution(speech_path: str, speech_id: Optional[List[str]], 
                             generated_rirs: List[np.ndarray], real_rirs: Optional[List[np.ndarray]],
                             save_path: str, sr: int, n_speech_files: int = 3):
    """Process speech files and convolve them with RIRs."""
    print(f"\n=== Processing Speech Convolution ===")
    
    # Find speech files
    speech_files = find_librispeech_files(speech_path, speech_id, n_speech_files)
    if not speech_files:
        print(f"XXXXXXX\nNo speech files found\nXXXXXXX")
        return
    print(f"Processing {len(speech_files)} speech files with {len(generated_rirs)} RIRs")
    
    # Create output directories
    convolved_dir = Path(save_path) / "convolved_speech"
    convolved_dir.mkdir(parents=True, exist_ok=True)
    
    clean_dir = convolved_dir / "clean"
    gen_dir = convolved_dir / "generated_rir"
    real_dir = convolved_dir / "real_rir"
    
    clean_dir.mkdir(exist_ok=True)
    gen_dir.mkdir(exist_ok=True)
    if real_rirs is not None:
        real_dir.mkdir(exist_ok=True)
    
    # Process each speech file
    for speech_idx, speech_file in enumerate(tqdm(speech_files, desc="Processing speech files")):
        # Load speech
        speech, speech_sr = load_speech_file(speech_file, target_sr=sr)
        
        if speech is None:
            continue
            
        file_name = Path(speech_file).stem
        
        # Save clean speech
        clean_path = clean_dir / f"clean_{speech_idx:02d}_{file_name}.wav"
        sf.write(clean_path, speech, sr)
        
        # Convolve with each generated RIR
        for rir_idx, gen_rir in enumerate(generated_rirs):
            try:
                reverb_speech = convolve_speech_with_rir(speech, gen_rir)
                output_path = gen_dir / f"gen_rir_{rir_idx:02d}_speech_{speech_idx:02d}_{file_name}.wav"
                sf.write(output_path, reverb_speech, sr)
            except Exception as e:
                print(f"Error convolving speech {speech_idx} with generated RIR {rir_idx}: {e}")
        
        # Convolve with each real RIR (if available)
        if real_rirs is not None:
            for rir_idx, real_rir in enumerate(real_rirs):
                try:
                    reverb_speech = convolve_speech_with_rir(speech, real_rir)
                    output_path = real_dir / f"real_rir_{rir_idx:02d}_speech_{speech_idx:02d}_{file_name}.wav"
                    sf.write(output_path, reverb_speech, sr)
                except Exception as e:
                    print(f"Error convolving speech {speech_idx} with real RIR {rir_idx}: {e}")
    
    print(f"Speech convolution completed!")
    print(f"Clean speech files: {clean_dir}")
    print(f"Generated RIR convolutions: {gen_dir}")
    if real_rirs is not None:
        print(f"Real RIR convolutions: {real_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate RIRs using trained diffusion model")
    # --- Cfg ---
    parser.add_argument("--mode", type=int, choices=[1, 2], default=2,
                    help="Mode 1: Random conditions, Mode 2: Dataset conditions with comparison")
    parser.add_argument('--nSamples', type=int, default=128, 
                        help="Only take effect for old runs where data_info['nSamples'] is not available. Number of samples to load from dataset.")
    parser.add_argument("--model_path", type=str,
                    default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/Jul02_19-01-47_dsief06/model_best.pth.tar',
                    help="Path to the trained model checkpoint (.pth.tar)")
    parser.add_argument("--save_path", type=str, default=None,
                    help="Directory to save generated plots and audio")
    parser.add_argument("--device", type=str, default=None,
                    help="Device to use (cuda/cpu). Auto-detect if not specified")
    parser.add_argument("--save_audio", action="store_true",
                    help="Save generated RIRs as audio files")
    parser.add_argument("--save_inference", action="store_true",
                    help="Save generated RIRs as audio files")
    parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible generation")
    parser.add_argument("--dataset_path", type=str, default=os.path.normpath('./datasets/GTU_rir/GTU_RIR.pickle.dat'),
                    help="Path to dataset (required for mode 2)")
    # Speech convolution arguments
    parser.add_argument("--speech_path", type=str, default='/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/1195/130164/', # /dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/1195/130164/
                    help="Path to LibriSpeech dataset for speech convolution")
    parser.add_argument("--speech_id", type=str, nargs='+', default=['1195-130164-0010','1195-130164-0010'],
                    help="Optional speaker ID to filter LibriSpeech files")
    parser.add_argument("--n_speech_files", type=int, default=2,
                    help="Number of speech files to process for convolution")
    # --- Parameters ---
    parser.add_argument("--n_timesteps", type=int, default=None,
                    help="None for the n_timesteps used in training, otherwise specify the number of timesteps for generation")
    parser.add_argument("--nRIR", type=int, default=5,
                    help="Number of RIRs to generate")
    parser.add_argument("--rir_indices", type=int, nargs='+', default=None,
                    help="Specific RIR indices to use from dataset (mode 2 only)")  
    # --- Dataset loading arguments (needed for mode 2) ---
    parser.add_argument("--use_spectrogram", type=bool, default=None,
                    help="Use spectrograms for generation (default from model config)")
    parser.add_argument("--hop_length", type=int, default=None,
                    help="Hop length for STFT")
    parser.add_argument("--n_fft", type=int, default=None,
                    help="FFT size for STFT")
    parser.add_argument("--sample_max_sec", type=float, default=None,
                    help="Maximum sample duration in seconds")
    parser.add_argument("--sr_target", type=int, default=None,
                    help="Target sample rate")
    parser.add_argument("--octaves", type=float, nargs='+', 
                    default=[125, 250, 500, 1000, 2000, 4000],
                    help="Octave center frequencies for EDC analysis")
    
    args = parser.parse_args()
    
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Running in mode {args.mode}")
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Create save directory
    if args.save_path is None:
        args.save_path = os.path.join(os.path.dirname(args.model_path), f"generated_rirs")
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    #  -------------- Load model and run configuration --------------
    model, config = load_model_and_config(args.model_path, device)
    data_info = config.get('data_info', {})
    sample_rate = data_info.get('sr_target', args.sr_target)
    n_fft = args.n_fft if args.n_fft is not None else data_info['n_fft']
    sample_max_sec = args.sample_max_sec if args.sample_max_sec is not None else data_info['sample_max_sec']
    hop_length = args.hop_length if args.hop_length is not None else data_info['hop_length']
    use_spectrogram = args.use_spectrogram if args.use_spectrogram is not None else data_info['use_spectrogram']
    sr_target = args.sr_target if args.sr_target is not None else data_info['sr_target']
    nSamples = data_info.get('nSamples', args.nSamples)
    scale_rir_enabled = data_info.get('scale_rir', False)
    if args.n_timesteps is not None:
        config['n_timesteps'] = args.n_timesteps
    print(f"Using sample rate: {sample_rate} Hz and {config['n_timesteps']} timesteps for generation")
    title = f"sr: {sr_target}Hz, nTimesteps: {config['n_timesteps']}\nhop_length: {hop_length}, n_fft: {n_fft}, sample_max_sec: {sample_max_sec}"
    # -------------- Generate or load conditions based on mode --------------
    if args.mode == 1:
        # Mode 1: Random conditions
        print("Mode 1: Generating random conditions...")
        conditions = generate_random_conditions(args.nRIR, device)
        real_rirs = None
        rir_indices = None
        
    else:
        # Mode 2: Load from dataset
        print("Mode 2: Loading conditions from dataset...")
        # Load dataset
        dataset = load_rir_dataset(
            'gtu', args.dataset_path, mode='raw', nSamples=nSamples,
            hop_length=hop_length, n_fft=n_fft, sr_target=sr_target,
            use_spectrogram=use_spectrogram, sample_max_sec=sample_max_sec
        )        
        # Load conditions and real RIRs
        conditions, real_rirs_wave, real_rirs_spec, rir_indices, k_factors = load_dataset_conditions(
            dataset, args.nRIR, data_info, use_spectrogram, args.rir_indices
        )
        conditions = conditions.to(device)
    
    
    # -------------- Generate RIRs --------------
    generated_specs, generated_rirs = generate_rirs_batch(model, conditions, device, config)
    
    print(f"Generated {len(generated_rirs)} RIR waveforms")
    
    # -------------- Create appropriate visualization --------------
    conditions_np = conditions.cpu().numpy()
    
    if args.mode == 1:
        create_base_plot_mode1(generated_rirs, conditions_np, sample_rate, args.save_path)
    else:
        # Convert generated spectrograms for plotting
        generated_rirs_spec_list = [generated_specs[i] for i in range(len(generated_rirs))]
        
        # Scaled-RIR
        if scale_rir_enabled:
            gen_rirs_wave_unscaled, gen_rirs_spec_unscaled, real_rirs_wave_scaled, real_rirs_spec_scaled = prepare_scaled_unscaled_datasets(
                real_rirs_wave, generated_rirs, k_factors, sr_target, n_fft, hop_length)
            full_save_path = os.path.join(args.save_path, 'rirs_comparison_mode2_scaled.png')
            # reverb speech
            create_base_plot_mode2(real_rirs_wave_scaled, generated_rirs, real_rirs_spec_scaled, generated_rirs_spec_list,
                        conditions_np, rir_indices, sample_rate, full_save_path, f"**Scaled Signal**\n{title}")
            full_save_path = os.path.join(args.save_path, 'edc_comparison_mode2_scaled.png')
            create_edc_plots_mode2(real_rirs_wave_scaled, generated_rirs, conditions_np, rir_indices, 
                        sample_rate, full_save_path, args.octaves, f"**Scaled Signal**\n{title}")
        else:
            gen_rirs_wave_unscaled = generated_rirs
            gen_rirs_spec_unscaled = generated_rirs_spec_list
            
        # Process speech convolution if speech_path is provided
        if args.speech_path:
            process_speech_convolution(
                args.speech_path, args.speech_id, 
                gen_rirs_wave_unscaled, real_rirs_wave,
                args.save_path, sample_rate, args.n_speech_files
            )
        return
        # plot unscaled signals
        create_base_plot_mode2(real_rirs_wave, gen_rirs_wave_unscaled, real_rirs_spec, gen_rirs_spec_unscaled,
                        conditions_np, rir_indices, sample_rate, args.save_path, title)
        create_edc_plots_mode2(real_rirs_wave, gen_rirs_wave_unscaled, conditions_np, rir_indices, 
                      sample_rate, args.save_path, args.octaves, title)

    
    # Save audio files if requested
    if args.save_audio:
        save_audio_files(generated_rirs, args.save_path, sample_rate, "generated")
        if args.mode == 2:
            save_audio_files(real_rirs, args.save_path, sample_rate, "real")
    
    # Save statistics
    if args.save_inference:
        save_generation_stats(generated_rirs, conditions_np, config, args.model_path, 
                            args.save_path, sample_rate, args.mode, real_rirs, rir_indices)
    
    print(f"\n Generation completed successfully!")
    print(f" Results saved to: {save_path}")
    print(f" Generated {args.nRIR} RIRs in mode {args.mode}")

    
if __name__ == "__main__":
    main()
