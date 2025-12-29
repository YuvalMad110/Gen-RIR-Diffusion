"""
Visualization utilities for RIR inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict


def format_condition_string(condition: np.ndarray, mode: str = "room") -> str:
    """Format condition parameters for display."""
    if mode == "room":
        room_dims = condition[:3]
        rt60 = condition[-1]
        return f"Room: {room_dims[0]:.1f}x{room_dims[1]:.1f}x{room_dims[2]:.1f}m, RT60: {rt60:.2f}s"
    elif mode == "locations":
        mic_loc = condition[3:6]
        speaker_loc = condition[6:9]
        return f"Mic: ({mic_loc[0]:.1f},{mic_loc[1]:.1f},{mic_loc[2]:.1f}), Src: ({speaker_loc[0]:.1f},{speaker_loc[1]:.1f},{speaker_loc[2]:.1f})"
    return ""


def plot_comparison(real_rirs_wave: List[np.ndarray], generated_rirs_wave: List[np.ndarray],
                    real_rirs_spec: List[np.ndarray], generated_rirs_spec: List[np.ndarray],
                    conditions: np.ndarray, rir_indices: List[int], sr: int, save_path: str,
                    title: str = "", metrics: Optional[Dict] = None):
    """Create comparison plot: real vs generated (waveforms and spectrograms)."""
    n_rirs = len(real_rirs_wave)
    fig, axes = plt.subplots(n_rirs, 4, figsize=(20, 4 * n_rirs))
    
    if title:
        fig.suptitle(title, fontsize=12)
    if n_rirs == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_rirs):
        idx = rir_indices[i]
        real_wave, gen_wave = real_rirs_wave[i], generated_rirs_wave[i]
        real_spec, gen_spec = real_rirs_spec[i], generated_rirs_spec[i]
        
        condition_str = format_condition_string(conditions[i], "room")
        location_str = format_condition_string(conditions[i], "locations")
        
        # Real waveform
        time_real = np.arange(len(real_wave)) / sr
        axes[i, 0].plot(time_real, real_wave, linewidth=0.8, color='green')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'Real RIR #{idx} - {condition_str}')
        
        # Generated waveform
        time_gen = np.arange(len(gen_wave)) / sr
        axes[i, 1].plot(time_gen, gen_wave, linewidth=0.8, color='blue')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title(f'Generated RIR #{idx}')
        
        # Real spectrogram
        real_mag = np.sqrt(real_spec[0]**2 + real_spec[1]**2)
        axes[i, 2].imshow(20 * np.log10(real_mag + 1e-8), aspect='auto', origin='lower', cmap='viridis')
        axes[i, 2].set_title(f'Real RIR #{idx} Spectrogram')
        axes[i, 2].set_ylabel('Frequency Bin')
        
        # Generated spectrogram
        gen_mag = np.sqrt(gen_spec[0]**2 + gen_spec[1]**2)
        axes[i, 3].imshow(20 * np.log10(gen_mag + 1e-8), aspect='auto', origin='lower', cmap='viridis')
        axes[i, 3].set_title(f'Generated RIR #{idx} Spectrogram - {location_str}')
        axes[i, 3].set_ylabel('Frequency Bin')
        
        if i == n_rirs - 1:
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 2].set_xlabel('Time Frame')
            axes[i, 3].set_xlabel('Time Frame')
    
    plt.tight_layout()
    plot_path = save_path if save_path.endswith(".png") else str(Path(save_path) / "rirs_comparison.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved to: {plot_path}")


def plot_edc_comparison(real_rirs: List[np.ndarray], generated_rirs: List[np.ndarray],
                        conditions: np.ndarray, rir_indices: List[int], sr: int, save_path: str,
                        metrics: Optional[Dict] = None, octave_bands: List[float] = None,
                        title: str = ""):
    """Create Energy Decay Curve comparison plot."""
    n_rirs = len(real_rirs)
    fig, axes = plt.subplots(n_rirs, 1, figsize=(12, 4 * n_rirs))
    
    if title:
        fig.suptitle(title, fontsize=12)
    if n_rirs == 1:
        axes = [axes]
    
    for i in range(n_rirs):
        idx = rir_indices[i]
        real_edc = _calculate_edc_numpy(real_rirs[i])
        gen_edc = _calculate_edc_numpy(generated_rirs[i])
        
        time_real = np.arange(len(real_edc)) / sr
        time_gen = np.arange(len(gen_edc)) / sr
        
        axes[i].plot(time_real, 10 * np.log10(real_edc + 1e-10), 
                     'g-', linewidth=1.5, label='Real', alpha=0.8)
        axes[i].plot(time_gen, 10 * np.log10(gen_edc + 1e-10), 
                     'b--', linewidth=1.5, label='Generated', alpha=0.8)
        
        rt60 = conditions[i, -1]
        axes[i].axhline(y=-60, color='r', linestyle=':', alpha=0.5, label='RT60 threshold')
        axes[i].axvline(x=rt60, color='orange', linestyle=':', alpha=0.5, label=f'RT60={rt60:.2f}s')
        
        condition_str = format_condition_string(conditions[i], "room")
        axes[i].set_title(f'EDC Comparison #{idx} - {condition_str}')
        axes[i].set_ylabel('Energy (dB)')
        axes[i].set_ylim(-80, 5)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')
        
        if i == n_rirs - 1:
            axes[i].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plot_path = save_path if save_path.endswith(".png") else str(Path(save_path) / "edc_comparison.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"EDC comparison plot saved to: {plot_path}")


def _calculate_edc_numpy(rir: np.ndarray) -> np.ndarray:
    """Calculate Energy Decay Curve from RIR."""
    energy = rir ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    return edc / (edc[0] + 1e-10)
