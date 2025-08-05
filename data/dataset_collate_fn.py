import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader
from utils.signal_proc import waveform_to_spectrogram, scale_rir

def scale_and_spectrogram_collate_fn(sr: float, db_cutoff: float = -40.0, 
                                   n_fft: int = 256, hop_length: int = 64, 
                                   scale_rir_flag: bool = True, use_spectrogram: bool = True,
                                   apply_zero_tail: bool = False):
    """
    Creates a collate function that optionally scales time-domain RIRs and/or converts to spectrograms.
    
    Args:
        sr: Sample rate
        db_cutoff: dB cutoff for EDC cropping
        n_fft: FFT size for spectrogram
        hop_length: Hop length for spectrogram
        scale_rir_flag: Whether to apply RIR scaling
        use_spectrogram: Whether to convert to spectrogram
        apply_zero_tail: Whether to zero out signal after db_cutoff point
    
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
            batch_rirs = scale_rir(batch_rirs, sr, db_cutoff, apply_zero_tail)
        
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