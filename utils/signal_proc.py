import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple

# ------------------------------------------------------------------------
#                       Signal Transformations
# ------------------------------------------------------------------------
def waveform_to_spectrogram(waveform, hop_length, n_fft, sr=None) -> tuple:
    """
    Convert waveform(s) to spectrogram using the same approach as the training dataset.
    
    Args:
        waveform: Waveform signal(s) - shape [B, T] for batch or [T] for single
        hop_length: Hop length for STFT
        n_fft: FFT size for STFT
        sr: Sample rate - if provided, will return frequency and time grids
        
    Returns:
        If sr not provided: Spectrogram of shape [B, 2, freq, time] or [2, freq, time]
        If sr provided: (spectrogram, freq_grid, time_grid)
    """
    # Convert numpy to torch tensor
    if isinstance(waveform, np.ndarray):
        rir_tensor = torch.from_numpy(waveform).float()
    else:
        rir_tensor = waveform.float()
    # Handle single waveform vs batch
    if rir_tensor.dim() == 1:
        # Single waveform - add batch dimension
        rir_tensor = rir_tensor.unsqueeze(0)  # [1, T]
        single_input = True
    elif rir_tensor.dim() == 2:
        # Batch of waveforms - shape [B, T]
        single_input = False
    else:
        raise ValueError(f"Expected 1D or 2D input, got {rir_tensor.dim()}D")
    
    batch_size = rir_tensor.shape[0]
    
    # Create Hann window (same as in training)
    window = torch.hann_window(n_fft, device=rir_tensor.device)
    # Apply STFT to each waveform in the batch
    batch_specs = []
    for i in range(batch_size):
        # Apply STFT (same as training)
        rir_stft = torch.stft(
            rir_tensor[i], 
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            window=window
        )
        # Stack real and imaginary parts (same as training)
        rir_spec = torch.stack((rir_stft.real, rir_stft.imag), dim=0)  # [2, F, T]
        batch_specs.append(rir_spec)
    
    # Stack all spectrograms
    batch_specs = torch.stack(batch_specs, dim=0)  # [B, 2, F, T]    
    # Remove batch dimension if single input
    if single_input:
        batch_specs = batch_specs.squeeze(0)  # [2, F, T]
    
    if sr:
        # Create frequency grid
        freq_grid = np.fft.fftfreq(n_fft, 1/sr)[:n_fft//2 + 1]  # Only positive frequencies
        # Create time grid
        n_frames = batch_specs.shape[-1]  # Last dimension is time
        time_grid = np.arange(n_frames) * hop_length / sr
        
        return batch_specs, freq_grid, time_grid
    else:
        return batch_specs

def spectrogram_to_waveform(spectrogram, hop_length, n_fft):
    """
    Convert spectrogram(s) back to waveform using the inverse of the training approach.
    
    Args:
        spectrogram: Complex spectrogram(s) - shape [B, 2, freq, time] for batch or [2, freq, time] for single
        hop_length: Hop length used in STFT
        n_fft: FFT size used in STFT
        
    Returns:
        Reconstructed waveform(s) - shape [B, T] for batch or [T] for single
    """
    # ------- Prepare inputs ---------
    # Convert numpy to torch tensor
    if isinstance(spectrogram, np.ndarray):
        spec_tensor = torch.from_numpy(spectrogram).float()
    else:
        spec_tensor = spectrogram.float()
    # Handle single spectrogram vs batch
    single_input = False
    if spec_tensor.dim() == 3:
        # Single spectrogram - add batch dimension
        spec_tensor = spec_tensor.unsqueeze(0)  # [1, 2, F, T]
        single_input = True
    elif spec_tensor.dim() == 4:
        # Batch of spectrograms - shape [B, 2, F, T]
        pass
    else:
        raise ValueError(f"Expected 3D or 4D input, got {spec_tensor.dim()}D")
    
    batch_size = spec_tensor.shape[0]
    
    # ------- Inverse STFT ---------
    # Create Hann window (same as in forward transform)
    window = torch.hann_window(n_fft, device=spec_tensor.device)
    # Convert each spectrogram in the batch
    batch_waveforms = []
    for i in range(batch_size):
        # Reconstruct complex spectrogram from real and imaginary parts
        complex_spec = torch.complex(spec_tensor[i, 0], spec_tensor[i, 1])  # [F, T]
        
        # Use torch.istft (inverse of torch.stft used in forward transform)
        try:
            waveform = torch.istft(
                complex_spec, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                window=window,
                return_complex=False
            )
        except Exception as e:
            print(f"Warning: torch.istft failed for sample {i} ({e}), using librosa fallback")
            # Fallback to librosa method
            complex_spec_np = complex_spec.cpu().numpy()
            waveform = librosa.istft(complex_spec_np, hop_length=hop_length, n_fft=n_fft)
            waveform = torch.from_numpy(waveform)
        
        batch_waveforms.append(waveform)
    
    # Stack all waveforms
    batch_waveforms = torch.stack(batch_waveforms, dim=0)  # [B, T]    
    # Remove batch dimension if single input
    if single_input:
        batch_waveforms = batch_waveforms.squeeze(0)  # [T]
    
    return batch_waveforms

def plot_waveform(signals, legend=None, title=None, sr=None, save_path=None):
    """
    Plot multiple waveform signals on a single graph.
    
    Args:
        signals: Single waveform, list of waveforms, or batch tensor/array [B, T] or [T]
        legend: List of legend labels. If None, uses indices (1, 2, ...)
        title: Plot title. If None, uses default title
        sr: Sample rate. If provided, x-axis shows time in seconds
        save_path: Path to save the plot. If None, only displays the plot
    """    
    # Convert to numpy if torch tensor
    if torch.is_tensor(signals):
        signals_np = signals.detach().cpu().numpy()
    else:
        signals_np = np.array(signals)
    
    # Handle different input formats
    if signals_np.ndim == 1:
        # Single signal - convert to 2D
        signals_np = signals_np.reshape(1, -1)
    elif signals_np.ndim == 2:
        # Multiple signals - assume [B, T] format
        pass
    else:
        raise ValueError(f"Expected 1D or 2D input, got {signals_np.ndim}D")
    
    n_signals = signals_np.shape[0]
    signal_length = signals_np.shape[1]
    
    # Create time axis
    if sr is not None:
        time_axis = np.arange(signal_length) / sr
        x_label = 'Time (s)'
    else:
        time_axis = np.arange(signal_length)
        x_label = 'Sample Index'
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot each signal
    for i in range(n_signals):
        plt.plot(time_axis, signals_np[i], linewidth=1.0, alpha=0.8)
    
    # Set up legend
    if legend is None:
        legend = [f'{i+1}' for i in range(n_signals)]
    elif len(legend) != n_signals:
        print(f"Warning: Legend length ({len(legend)}) doesn't match number of signals ({n_signals})")
        legend = [f'{i+1}' for i in range(n_signals)]
    
    if n_signals > 1:  # Only show legend if multiple signals
        plt.legend(legend)
    
    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')
    
    if title is None:
        if n_signals == 1:
            plt.title('Waveform')
        else:
            plt.title(f'Waveforms ({n_signals} signals)')
    else:
        plt.title(title)
    
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show the plot (this will open in VS Code)
    plt.show()

# ------------------------------------------------------------------------
#                       Signal Scaling
# ------------------------------------------------------------------------
def calculate_edc(rir: torch.Tensor) -> torch.Tensor:
    """
    Calculate Energy Decay Curve (EDC) from time-domain RIR.
    
    Args:
        rir: Time-domain RIR tensor [batch, time]
                       
                                             
    
    Returns:
        EDC tensor normalized to [0, 1]
    """
                           
    energy = rir ** 2
    edc = torch.cumsum(torch.flip(energy, dims=[-1]), dim=-1)
    edc = torch.flip(edc, dims=[-1])
    edc = edc / (torch.max(edc, dim=-1, keepdim=True)[0] + 1e-12)
    return edc

def estimate_decay_k_factor(edc: torch.Tensor, sr: float, db_cutoff: float = -40.0) -> torch.Tensor:
    """
    Estimate exponential decay rate k from EDC using least squares fit.
    
    Args:
        edc: Energy Decay Curve tensor [batch, time]
        sr: Sample rate
        db_cutoff: dB cutoff for EDC cropping
    
    Returns:
        Decay rate k for each sample in batch [batch]
    """
    # Crop EDC by dB cutoff
    edc_db = 10 * torch.log10(edc + 1e-12)
    
    # Find crop index for each sample in batch
    crop_indices = []
    for i in range(edc.shape[0]):
        cutoff_mask = edc_db[i] < db_cutoff
        if cutoff_mask.any():
            crop_idx = torch.where(cutoff_mask)[0][0].item()
        else:
            crop_idx = edc.shape[1]  # Use full length if no cutoff found
        crop_indices.append(crop_idx)
    
    # Crop to minimum index to keep batch uniform
    min_crop_idx = min(crop_indices)
    cropped_edc = edc[:, :min_crop_idx]
    
    # Estimate decay rate
    batch_size, seq_len = cropped_edc.shape
    t = torch.arange(seq_len, dtype=torch.float32, device=cropped_edc.device) / sr
    log_edc = torch.log(cropped_edc + 1e-12)
    
    # Least squares fitting: log_edc = slope * t + intercept
    A = torch.stack([t, torch.ones_like(t)], dim=0).T
    A = A.unsqueeze(0).expand(batch_size, -1, -1)
    
    AtA = torch.bmm(A.transpose(-2, -1), A)
    Atb = torch.bmm(A.transpose(-2, -1), log_edc.unsqueeze(-1))
    
    solution = torch.linalg.solve(AtA, Atb)
    slope = solution[:, 0, 0]
    k = -slope / 2
    
    return k

def apply_rir_scaling(rir: torch.Tensor, k: torch.Tensor, sr: float) -> torch.Tensor:
    """
    Apply exponential scaling to RIR using provided k factors.
    
    Args:
        rir: Time-domain RIR tensor [batch, time]
        k: Decay rate factors [batch]
        sr: Sample rate
    
    Returns:
        Scaled RIR tensor
    """
    rir_seq_len = rir.shape[-1]
    t_full = torch.arange(rir_seq_len, dtype=torch.float32, device=rir.device) / sr
    scaling_factor = torch.exp(k.unsqueeze(-1) * t_full.unsqueeze(0))
    
    return rir * scaling_factor

def undo_rir_scaling(scaled_rir: torch.Tensor, k: torch.Tensor, sr: float) -> torch.Tensor:
    """
    Undo exponential scaling from RIR using provided k factors.
    
    Args:
        scaled_rir: Scaled time-domain RIR tensor [batch, time]
        k: Decay rate factors used for original scaling [batch]
        sr: Sample rate
    
    Returns:
        Unscaled RIR tensor
    """
    rir_seq_len = scaled_rir.shape[-1]
    t_full = torch.arange(rir_seq_len, dtype=torch.float32, device=scaled_rir.device) / sr
    # Inverse scaling: divide by the scaling factor (or multiply by its inverse)
    inverse_scaling_factor = torch.exp(-k.unsqueeze(-1) * t_full.unsqueeze(0))
    return scaled_rir * inverse_scaling_factor

def scale_rir(rir: torch.Tensor, sr: float, db_cutoff: float = -40.0) -> torch.Tensor:
    """
    Scale RIR by estimating and compensating for natural decay.
    
    Args:
        rir: Time-domain RIR tensor [batch, time]
        sr: Sample rate
        db_cutoff: dB cutoff for EDC cropping
    
    Returns:
        Scaled RIR tensor
    """
    # Calculate EDC
    edc = calculate_edc(rir)
    
    # Estimate decay rate k
    k = estimate_decay_k_factor(edc, sr, db_cutoff)
    
    # Apply scaling
# ------------------------------------------------------------------------
#                       Miscellaneous
# ------------------------------------------------------------------------
def normalize_signals(signals: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize a list of RIRs to unit maximum amplitude.
    
    Args:
        signals: List of RIR waveforms
        
    Returns:
        List of normalized RIR waveforms
    """
    normalized_signals = []
    
    for i, sig in enumerate(signals):
        if np.max(np.abs(sig)) > 0:
            norm_sig = sig / np.max(np.abs(sig))
        else:
            # Handle case where RIR is all zeros
            norm_sig = sig.copy()
            
        normalized_signals.append(norm_sig)
    
    return normalized_signals