import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader
from utils.signal_proc import waveform_to_spectrogram, scale_rir

# def convert_to_spectrogram(signals: torch.Tensor, n_fft: int = 512, hop_length: int = 256) -> torch.Tensor:
#     """Convert batch of time-domain signals to spectrograms."""
#     batch_size = signals.shape[0]
#     spectrograms = []
    
#     for i in range(batch_size):
#         # Remove channel dimension if present: [1, T] -> [T]
#         signal = signals[i].squeeze(0) if signals[i].dim() > 1 else signals[i]
        
#         # Create window on same device as signal
#         window = torch.hann_window(n_fft, device=signal.device)
        
#         # Convert to spectrogram
#         spec_complex = torch.stft(
#             signal, 
#             n_fft=n_fft, 
#             hop_length=hop_length, 
#             window=window,
#             return_complex=True
#         )
        
#         # Stack real and imaginary parts: [2, freq, time]
#         spec = torch.stack([spec_complex.real, spec_complex.imag], dim=0)
#         spectrograms.append(spec)
    
#     return torch.stack(spectrograms)  # [batch, 2, freq, time]

def scale_and_spectrogram_collate_fn(sr: float, db_cutoff: float = -40.0, 
                                   n_fft: int = 256, hop_length: int = 64, 
                                   scale_rir_flag: bool = True, use_spectrogram: bool = True):
    """
    Creates a collate function that optionally scales time-domain RIRs and/or converts to spectrograms.
    
    Args:
        sr: Sample rate
        db_cutoff: dB cutoff for EDC cropping
        n_fft: FFT size for spectrogram
        hop_length: Hop length for spectrogram
        scale_rir_flag: Whether to apply RIR scaling
        use_spectrogram: Whether to convert to spectrogram
    
    Returns:
        Collate function for DataLoader
    """
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        rirs, room_dims, mic_locs, speaker_locs, rt60s = zip(*batch)
        
        # Stack time-domain RIRs and move to device
        batch_rirs = torch.stack(rirs)
        
        # Handle different input shapes
        if batch_rirs.dim() == 3:  # [batch, channels, time]
            # Take first channel for scaling (assuming mono or using first channel)
            batch_rirs = batch_rirs[:, 0, :] # [batch, time]
        
        # Optionally scale the RIRs
        if scale_rir_flag:
            batch_rirs = scale_rir(batch_rirs, sr, db_cutoff)
        
        # Optionally convert to spectrograms
        if use_spectrogram:
            batch_rirs = waveform_to_spectrogram(waveform=batch_rirs, hop_length=hop_length, n_fft=n_fft)
        
        # Stack other tensors and move to device
        batch_room_dims = torch.stack([torch.tensor(rd, dtype=torch.float32) for rd in room_dims])
        batch_mic_locs = torch.stack([torch.tensor(ml, dtype=torch.float32) for ml in mic_locs])
        batch_speaker_locs = torch.stack([torch.tensor(sl, dtype=torch.float32) for sl in speaker_locs])
        batch_rt60s = torch.tensor(rt60s, dtype=torch.float32)
        return batch_rirs, batch_room_dims, batch_mic_locs, batch_speaker_locs, batch_rt60s
    
    return collate_fn