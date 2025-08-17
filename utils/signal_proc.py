import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
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

def estimate_decay_k_factor(edc: torch.Tensor, sr: float, db_cutoff: float = -40.0) -> Tuple[torch.Tensor, List[int]]:
    """
    Estimate exponential decay rate k from EDC using least squares fit.
    
    Args:
        edc: Energy Decay Curve tensor [batch, time]
        sr: Sample rate
        db_cutoff: dB cutoff for EDC cropping
    
    Returns:
        Tuple of (decay rate k for each sample [batch], crop indices for each sample [batch])
    """
    # Convert to dB
    edc_db = 10 * torch.log10(edc + 1e-12)
    
    batch_size = edc.shape[0]
    k_values = []
    crop_indices = []
    
    # Pre-generate full time vector
    t_full = torch.arange(edc.shape[1], dtype=torch.float32, device=edc.device) / sr
    
    # Process each sample individually with its own cutoff
    for i in range(batch_size):
        # Find crop index for this specific sample
        cutoff_mask = edc_db[i] < db_cutoff
        if cutoff_mask.any():
            crop_idx = torch.where(cutoff_mask)[0][0].item()
        else:
            crop_idx = edc.shape[1]  # Use full length if no cutoff found
        
        crop_indices.append(crop_idx)
        
        # Crop this sample's EDC and time vector
        cropped_edc = edc[i, :crop_idx]
        t = t_full[:crop_idx]
        log_edc = torch.log(cropped_edc + 1e-12)
        
        # Least squares fitting: log_edc = slope * t + intercept
        A = torch.stack([t, torch.ones_like(t)], dim=1)  # [seq_len, 2]
        
        # Normal equation: (A^T A)^-1 A^T b
        AtA = torch.mm(A.T, A)  # [2, 2]
        Atb = torch.mv(A.T, log_edc)  # [2]
        solution = torch.linalg.solve(AtA, Atb)  # [2]
        slope = solution[0]
        
        k = -slope / 2
        k_values.append(k)
    
    return torch.stack(k_values), crop_indices

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

def scale_rir(rir: torch.Tensor, sr: float, db_cutoff: float = -40.0, apply_zero_tail: bool = False) -> torch.Tensor:
    """
    Scale RIR by estimating and compensating for natural decay.
    
    Args:
        rir: Time-domain RIR tensor [batch, time]
        sr: Sample rate
        db_cutoff: dB cutoff for EDC cropping
        apply_zero_tail: Whether to zero out signal after db_cutoff point
    
    Returns:
        Scaled RIR tensor
    """
    # Calculate EDC
    edc = calculate_edc(rir)
    
    # Estimate decay rate k and get crop indices
    k, crop_indices = estimate_decay_k_factor(edc, sr, db_cutoff)
    
    # Apply scaling
    scaled_rir = apply_rir_scaling(rir, k, sr)
    
    # Apply zero tail if requested
    if apply_zero_tail:
        for i, crop_idx in enumerate(crop_indices):
            scaled_rir[i, crop_idx:] = 0.0
    
    return scaled_rir

# ------------------------------------------------------------------------
#                       Signal Distance Metrics
# ------------------------------------------------------------------------
def cosine_distance_rir(generated_rir: List[np.ndarray], 
                       real_rir: List[np.ndarray], force_same_length=False) -> Tuple[List[float], float]:
    """
    Calculate Cosine Distance (CD) between generated and real RIRs.
    
    Formula: CD(H, Ĥ) = (1/M) * Σ(1 - (h_i^T * ĥ_i) / (||h_i|| * ||ĥ_i||))
    
    Args:
        generated_rir: List of generated RIR arrays, each shaped (1, rir_len)
        real_rir: List of real RIR arrays, each shaped (1, rir_len)
        force_same_length: If True, will truncate longer RIRs to match the shorter ones
    
    Returns:
        Tuple containing:
        - List of individual cosine distances for each RIR pair
        - Total average cosine distance across all pairs
    """
    
    if len(generated_rir) != len(real_rir):
        raise ValueError("Generated and real RIR lists must have the same length")
    
    if len(generated_rir) == 0:
        raise ValueError("RIR lists cannot be empty")
    
    individual_distances = []
    
    for i, (gen_rir, real_rir_i) in enumerate(zip(generated_rir, real_rir)):
        # Ensure arrays are shaped correctly (flatten if needed)
        if gen_rir.ndim == 2:
            gen_rir = gen_rir.flatten()
        if real_rir_i.ndim == 2:
            real_rir_i = real_rir_i.flatten()

        if len(gen_rir) != len(real_rir_i):
            if force_same_length:
                min_length = min(len(gen_rir), len(real_rir_i))
                gen_rir = gen_rir[:min_length]
                real_rir_i = real_rir_i[:min_length]
            else:
                raise ValueError(f"RIR pair {i} has mismatched lengths: {len(gen_rir)} vs {len(real_rir_i)}")
            
        if len(gen_rir) != len(real_rir_i):
            raise ValueError(f"RIR pair {i} has mismatched lengths: "
                           f"{len(gen_rir)} vs {len(real_rir_i)}")
        
        # Calculate dot product (h_i^T * ĥ_i)
        dot_product = np.dot(real_rir_i, gen_rir)
        
        # Calculate L2 norms (||h_i|| and ||ĥ_i||)
        norm_real = np.linalg.norm(real_rir_i)
        norm_gen = np.linalg.norm(gen_rir)
        
        # Handle edge case where one or both norms are zero
        if norm_real == 0 or norm_gen == 0:
            if norm_real == 0 and norm_gen == 0:
                cosine_similarity = 1.0  # Both zero vectors are "similar"
            else:
                cosine_similarity = 0.0  # One zero, one non-zero
        else:
            # Calculate cosine similarity
            cosine_similarity = dot_product / (norm_real * norm_gen)
            # Clamp to [-1, 1] to handle numerical precision issues
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        # Calculate cosine distance (1 - cosine_similarity)
        cosine_dist = 1.0 - cosine_similarity
        individual_distances.append(cosine_dist)
    
    # Calculate total average distance
    total_distance = np.mean(individual_distances)
    
    return individual_distances, total_distance

def nmse_rir(generated_rir: List[np.ndarray], 
             real_rir: List[np.ndarray], force_same_length=False) -> Tuple[List[float], float]:
    """
    Calculate Normalized Mean Square Error (NMSE) between generated and real RIRs.
    
    Formula: NMSE(H, Ĥ) = (1/M) * Σ(||ĥᵢ - hᵢ||² / ||hᵢ||²)
    
    Args:
        generated_rir: List of generated RIR arrays, each shaped (1, rir_len)
        real_rir: List of real RIR arrays, each shaped (1, rir_len)
        force_same_length: If True, will truncate longer RIRs to match the shorter ones
    
    Returns:
        Tuple containing:
        - List of individual NMSE values for each RIR pair
        - Total average NMSE across all pairs
    """
    
    if len(generated_rir) != len(real_rir):
        raise ValueError("Generated and real RIR lists must have the same length")
    
    if len(generated_rir) == 0:
        raise ValueError("RIR lists cannot be empty")
    
    individual_nmse = []
    
    for i, (gen_rir, real_rir_i) in enumerate(zip(generated_rir, real_rir)):

            
        # Ensure arrays are shaped correctly (flatten if needed)
        if gen_rir.ndim == 2:
            gen_rir = gen_rir.flatten()
        if real_rir_i.ndim == 2:
            real_rir_i = real_rir_i.flatten()

        if len(gen_rir) != len(real_rir_i):
            if force_same_length:
                min_length = min(len(gen_rir), len(real_rir_i))
                gen_rir = gen_rir[:min_length]
                real_rir_i = real_rir_i[:min_length]
            else:
                raise ValueError(f"RIR pair {i} has mismatched lengths: {len(gen_rir)} vs {len(real_rir_i)}")
            
               
        # Calculate difference (ĥᵢ - hᵢ)
        difference = gen_rir - real_rir_i
        
        # Calculate squared L2 norm of difference ||ĥᵢ - hᵢ||²
        numerator = np.sum(difference ** 2)
        
        # Calculate squared L2 norm of real RIR ||hᵢ||²
        denominator = np.sum(real_rir_i ** 2)
        
        # Handle edge case where real RIR has zero energy
        if denominator == 0:
            if numerator == 0:
                # Both are zero, perfect match
                nmse_value = 0.0
            else:
                # Real is zero but generated is not, infinite error
                # We'll use a large value instead of infinity
                nmse_value = float('inf')
                print(f"Warning: Real RIR {i} has zero energy, NMSE is infinite")
        else:
            # Calculate NMSE for this pair
            nmse_value = numerator / denominator
        
        individual_nmse.append(nmse_value)
    
    # Calculate total average NMSE (excluding infinite values)
    finite_nmse = [x for x in individual_nmse if np.isfinite(x)]
    if len(finite_nmse) == 0:
        total_nmse = float('inf')
    else:
        total_nmse = np.mean(finite_nmse)
    
    return individual_nmse, total_nmse

def evaluate_rir_quality(generated_rir: List[np.ndarray], 
                        real_rir: List[np.ndarray], force_same_length=False) -> dict:
    """
    Calling for multiple RIR quality metrics and returns a dictionary with the results

    """
    # Calculate NMSE
    individual_nmse, total_nmse = nmse_rir(generated_rir, real_rir, force_same_length)
    # Calculate Cosine Distance
    individual_cd, total_cd = cosine_distance_rir(generated_rir, real_rir, force_same_length)
    metrics = {
        'nmse': {
            'individual': individual_nmse,
            'total': total_nmse
        },
        'cosine_distance': {
            'individual': individual_cd,
            'total': total_cd
        }
    }
    return metrics

def sisdr_metric(signals_A: List[np.ndarray], signals_B: List[np.ndarray], force_same_length=False) -> Tuple[List[float], float]:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between signals_A to signals_B.
    
    SI-SDR is defined as:
    SI-SDR = 10 * log10(||s_target||² / ||e_noise||²)
    
    where:
    - s_target = <ĥ, h> / ||h||² * h  (target signal component)
    - e_noise = ĥ - s_target  (noise/distortion component)
    - ĥ is signals_A, h is signals_B
    
    Args:
        signals_A: List of arrays, each shaped (1, sig_len)
        signals_B: List of arrays, each shaped (1, sig_len)
        force_same_length: If True, will truncate longer RIRs to match the shorter ones
    
    Returns:
        Tuple containing:
        - List of individual SI-SDR values for each signal pair (in dB)
        - Total average SI-SDR across all pairs (in dB)
    """
    
    if len(signals_A) != len(signals_B):
        raise ValueError("Generated and real RIR lists must have the same length")
    
    individual_sisdr = []
    
    for i, (sig_a, sig_b) in enumerate(zip(signals_A, signals_B)):
        # Ensure arrays are shaped correctly (flatten if needed)
        if sig_a.ndim == 2:
            sig_a = sig_a.flatten()
        if sig_b.ndim == 2:
            sig_b = sig_b.flatten()
        # check signlas lengths
        if len(sig_a) != len(sig_b):
            if force_same_length:
                min_length = min(len(sig_a), len(sig_b))
                sig_a = sig_a[:min_length]
                sig_b = sig_b[:min_length]
            else:
                raise ValueError(f"Signals pair {i} has mismatched lengths: {len(sig_a)} vs {len(sig_b)}")
        
        # Calculate the scaling factor: <ĥ, h> / ||h||²
        dot_product = np.dot(sig_a, sig_b)
        real_energy = np.sum(sig_b ** 2)
        
        # Handle edge case where real RIR has zero energy
        if real_energy == 0:
            if np.sum(sig_a ** 2) == 0:
                # Both are zero, perfect match
                sisdr_value = float('inf')
            else:
                # Real is zero but generated is not, worst case
                sisdr_value = float('-inf')
                print(f"Warning: Real RIR {i} has zero energy, SI-SDR is -inf")
        else:
            # Calculate scaling factor
            alpha = dot_product / real_energy
            
            # Calculate target signal: s_target = α * h
            s_target = alpha * sig_b
            
            # Calculate noise/distortion: e_noise = ĥ - s_target
            e_noise = sig_a - s_target
            
            # Calculate energies
            s_target_energy = np.sum(s_target ** 2)
            e_noise_energy = np.sum(e_noise ** 2)
            
            # Calculate SI-SDR in dB
            if e_noise_energy == 0:
                # Perfect reconstruction
                sisdr_value = float('inf')
            else:
                sisdr_value = 10 * np.log10(s_target_energy / e_noise_energy)
        
        individual_sisdr.append(sisdr_value)
    
    # Calculate total average SI-SDR (excluding infinite values)
    finite_sisdr = [x for x in individual_sisdr if np.isfinite(x)]
    if len(finite_sisdr) == 0:
        if all(x == float('inf') for x in individual_sisdr):
            total_sisdr = float('inf')
        else:
            total_sisdr = float('-inf')
    else:
        total_sisdr = np.mean(finite_sisdr)
    
    return individual_sisdr, total_sisdr


def save_metrics_txt(metrics, rir_indices, file_path, conditions=None):
    """Save metrics as a human-readable text summary."""

    file_path = os.path.join(file_path, 'metrics_sum.txt') if not file_path.endswith('.txt') else file_path
    with open(file_path, 'w') as f:
        f.write("RIR EVALUATION METRICS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        for metric_name, metric_values in metrics.items():
            total_val = metric_values['total']
            if np.isfinite(total_val):
                if metric_name.lower() == 'sisdr':
                    f.write(f"{metric_name.upper()}: {total_val:.2f} dB\n")
                else:
                    f.write(f"{metric_name.upper()}: {total_val:.6f}\n")
            else:
                f.write(f"{metric_name.upper()}: {total_val}\n")
        
        f.write(f"\nNumber of RIR pairs evaluated: {len(rir_indices)}\n\n")
        
        # Individual results
        f.write("INDIVIDUAL RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'RIR':<6}")
        
        # Add condition headers if available
        if conditions is not None:
            f.write(f"{'Room (W×L×H)':<15}")
            if conditions.shape[1] > 3:
                f.write(f"{'RT60':<8}")
        
        # Add metric headers
        for metric_name in metrics.keys():
            if metric_name.lower() == 'sisdr':
                f.write(f"{metric_name.upper():<10}")
            else:
                f.write(f"{metric_name.upper():<12}")
        f.write("\n")
        
        # Data rows
        for i in range(len(rir_indices)):
            f.write(f"{rir_indices[i]:<6}")
            
            # Add condition data if available
            if conditions is not None:
                room_str = f"{conditions[i,0]:.1f}×{conditions[i,1]:.1f}×{conditions[i,2]:.1f}"
                f.write(f"{room_str:<15}")
                if conditions.shape[1] > 3:
                    f.write(f"{conditions[i,-1]:.2f}s{'':<3}")
            
            # Add metric values directly from metrics dictionary
            for metric_name in metrics.keys():
                val = metrics[metric_name]['individual'][i]
                if np.isfinite(val):
                    if metric_name.lower() == 'sisdr':
                        f.write(f"{val:8.2f}dB")
                    else:
                        f.write(f"{val:10.6f}  ")
                else:
                    if metric_name.lower() == 'sisdr':
                        f.write(f"{str(val):>9s} ")
                    else:
                        f.write(f"{str(val):>11s} ")
            f.write("\n")
        
        # Interpretation guide
        f.write(f"\n\nINTERPRETATION GUIDE:\n")
        f.write("-" * 20 + "\n")
        f.write("NMSE (Normalized Mean Square Error):\n")
        f.write("  - Lower values = better similarity\n")
        f.write("  - 0 = perfect match\n\n")
        f.write("Cosine Distance:\n")
        f.write("  - Lower values = better similarity\n") 
        f.write("  - 0 = identical direction\n\n")
        f.write("SI-SDR (Scale-Invariant Signal-to-Distortion Ratio):\n")
        f.write("  - Higher values = better quality\n")
        f.write("  - >20dB = excellent, 10-20dB = good, 0-10dB = fair, <0dB = poor\n")
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