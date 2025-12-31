"""
Acoustic Metrics for RIR Evaluation

Comprehensive metrics for comparing generated RIRs against ground truth:
- T60 (Reverberation Time) - per octave band
- EDT (Early Decay Time)
- DRR (Direct-to-Reverberant Ratio)
- C50/C80 (Clarity)
- LSD (Log-Spectral Distance)
- EDC Distance (Energy Decay Curve)

Reference: ISO 3382-1:2009 for room acoustic parameters
"""

import numpy as np
from scipy import signal
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional, Union
import warnings


# =============================================================================
# Energy Decay Curve (EDC) - Foundation for T60/EDT
# =============================================================================

def compute_edc(rir: np.ndarray, sr: int = None) -> np.ndarray:
    """
    Compute Energy Decay Curve using Schroeder backward integration.
    
    EDC(t) = integral from t to inf of h^2(tau) dtau
    
    Args:
        rir: Room impulse response (1D array)
        sr: Sample rate (optional, for time axis)
    
    Returns:
        EDC in dB, normalized to 0 dB at start
    """
    rir = np.asarray(rir).flatten()
    
    # Schroeder backward integration
    squared = rir ** 2
    edc = np.cumsum(squared[::-1])[::-1]
    
    # Avoid log(0)
    edc = np.maximum(edc, np.finfo(float).eps)
    
    # Convert to dB, normalize to 0 at start
    edc_db = 10 * np.log10(edc / edc[0])
    
    return edc_db


def compute_edc_octave_bands(rir: np.ndarray, sr: int, 
                              center_freqs: List[float] = None) -> Dict[float, np.ndarray]:
    """
    Compute EDC for each octave band.
    
    Args:
        rir: Room impulse response
        sr: Sample rate
        center_freqs: List of octave band center frequencies
                     Default: [125, 250, 500, 1000, 2000, 4000] Hz
    
    Returns:
        Dictionary mapping center frequency to EDC (in dB)
    """
    if center_freqs is None:
        center_freqs = [125, 250, 500, 1000, 2000, 4000]
    
    rir = np.asarray(rir).flatten()
    edcs = {}
    
    for fc in center_freqs:
        # Octave band: fc/sqrt(2) to fc*sqrt(2)
        f_low = fc / np.sqrt(2)
        f_high = fc * np.sqrt(2)
        
        # Ensure we don't exceed Nyquist
        nyquist = sr / 2
        if f_low >= nyquist:
            continue
        f_high = min(f_high, nyquist * 0.99)
        
        # Butterworth bandpass filter
        try:
            sos = signal.butter(4, [f_low, f_high], btype='band', fs=sr, output='sos')
            filtered = signal.sosfilt(sos, rir)
            edcs[fc] = compute_edc(filtered)
        except Exception as e:
            warnings.warn(f"Could not compute EDC for {fc}Hz: {e}")
            continue
    
    return edcs


# =============================================================================
# T60 (Reverberation Time)
# =============================================================================

def estimate_t60_from_edc(edc_db: np.ndarray, sr: int, 
                          fit_range: Tuple[float, float] = (-5, -35)) -> float:
    """
    Estimate T60 from EDC using linear regression.
    
    Uses T30 method: fit line from -5dB to -35dB, extrapolate to -60dB.
    
    Args:
        edc_db: Energy decay curve in dB
        sr: Sample rate
        fit_range: (start_db, end_db) for linear fit
    
    Returns:
        T60 in seconds, or np.nan if estimation fails
    """
    start_db, end_db = fit_range
    
    # Find indices for fit range
    idx_start = np.argmax(edc_db <= start_db)
    idx_end = np.argmax(edc_db <= end_db)
    
    if idx_start >= idx_end or idx_end - idx_start < 10:
        return np.nan
    
    # Time axis
    t = np.arange(len(edc_db)) / sr
    
    # Linear regression
    try:
        slope, intercept, _, _, _ = linregress(t[idx_start:idx_end], edc_db[idx_start:idx_end])
        
        if slope >= 0:  # EDC should decay
            return np.nan
        
        # T60 = time for 60dB decay
        t60 = -60 / slope
        
        # Sanity check
        if t60 < 0 or t60 > 20:  # Unrealistic T60
            return np.nan
            
        return t60
    except Exception:
        return np.nan


def compute_t60(rir: np.ndarray, sr: int) -> float:
    """
    Compute broadband T60 from RIR.
    
    Args:
        rir: Room impulse response
        sr: Sample rate
    
    Returns:
        T60 in seconds
    """
    edc_db = compute_edc(rir)
    return estimate_t60_from_edc(edc_db, sr)


def compute_t60_octave_bands(rir: np.ndarray, sr: int,
                              center_freqs: List[float] = None) -> Dict[float, float]:
    """
    Compute T60 for each octave band.
    
    Args:
        rir: Room impulse response
        sr: Sample rate
        center_freqs: Octave band center frequencies
    
    Returns:
        Dictionary mapping center frequency to T60 (seconds)
    """
    edcs = compute_edc_octave_bands(rir, sr, center_freqs)
    t60s = {}
    
    for fc, edc_db in edcs.items():
        t60s[fc] = estimate_t60_from_edc(edc_db, sr)
    
    return t60s


def t60_error(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
              center_freqs: List[float] = None) -> Dict[str, Union[float, Dict[float, float]]]:
    """
    Compute T60 error between generated and reference RIR.
    
    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        sr: Sample rate
        center_freqs: Octave band center frequencies
    
    Returns:
        Dictionary with:
        - 'broadband': Absolute T60 error (seconds)
        - 'per_band': Dict of per-band T60 errors
        - 'mean_band_error': Mean absolute error across bands
    """
    # Broadband
    t60_gen = compute_t60(rir_gen, sr)
    t60_ref = compute_t60(rir_ref, sr)
    
    # Per octave band
    t60_gen_bands = compute_t60_octave_bands(rir_gen, sr, center_freqs)
    t60_ref_bands = compute_t60_octave_bands(rir_ref, sr, center_freqs)
    
    band_errors = {}
    for fc in t60_gen_bands:
        if fc in t60_ref_bands:
            if not np.isnan(t60_gen_bands[fc]) and not np.isnan(t60_ref_bands[fc]):
                band_errors[fc] = t60_gen_bands[fc] - t60_ref_bands[fc]
    
    valid_errors = [e for e in band_errors.values() if not np.isnan(e)]
    mean_error = np.mean(np.abs(valid_errors)) if valid_errors else np.nan
    
    return {
        'broadband': t60_gen - t60_ref if not (np.isnan(t60_gen) or np.isnan(t60_ref)) else np.nan,
        'broadband_gen': t60_gen,
        'broadband_ref': t60_ref,
        'per_band': band_errors,
        'mean_band_abs_error': mean_error
    }


# =============================================================================
# EDT (Early Decay Time)
# =============================================================================

def compute_edt(rir: np.ndarray, sr: int) -> float:
    """
    Compute Early Decay Time (EDT).
    
    EDT is based on the first 10dB of decay, extrapolated to 60dB.
    More perceptually relevant than T60 for subjective reverberance.
    
    Args:
        rir: Room impulse response
        sr: Sample rate
    
    Returns:
        EDT in seconds
    """
    edc_db = compute_edc(rir)
    return estimate_t60_from_edc(edc_db, sr, fit_range=(0, -10))


def edt_error(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute EDT error between generated and reference RIR."""
    edt_gen = compute_edt(rir_gen, sr)
    edt_ref = compute_edt(rir_ref, sr)
    
    return {
        'error': edt_gen - edt_ref if not (np.isnan(edt_gen) or np.isnan(edt_ref)) else np.nan,
        'generated': edt_gen,
        'reference': edt_ref
    }


# =============================================================================
# DRR (Direct-to-Reverberant Ratio)
# =============================================================================

def find_direct_sound_idx(rir: np.ndarray, threshold_db: float = -20) -> int:
    """
    Find the index of the direct sound (first significant peak).
    
    Args:
        rir: Room impulse response
        threshold_db: Threshold below peak to define direct sound region
    
    Returns:
        Index of direct sound peak
    """
    rir = np.asarray(rir).flatten()
    rir_abs = np.abs(rir)
    
    # Find absolute maximum
    peak_idx = np.argmax(rir_abs)
    
    return peak_idx


def compute_drr(rir: np.ndarray, sr: int, direct_window_ms: float = 2.5) -> float:
    """
    Compute Direct-to-Reverberant Ratio (DRR).
    
    DRR = 10 * log10(E_direct / E_reverberant)
    
    Args:
        rir: Room impulse response
        sr: Sample rate
        direct_window_ms: Window around direct sound (typically 2.5ms)
    
    Returns:
        DRR in dB
    """
    rir = np.asarray(rir).flatten()
    
    # Find direct sound
    direct_idx = find_direct_sound_idx(rir)
    
    # Direct sound window (typically ±1.25ms around peak)
    window_samples = int(direct_window_ms * sr / 1000)
    start_idx = max(0, direct_idx - window_samples // 2)
    end_idx = min(len(rir), direct_idx + window_samples // 2)
    
    # Energy in direct and reverberant parts
    e_direct = np.sum(rir[start_idx:end_idx] ** 2)
    e_reverb = np.sum(rir[end_idx:] ** 2)
    
    if e_reverb < np.finfo(float).eps:
        return np.inf
    
    drr = 10 * np.log10(e_direct / e_reverb)
    
    return drr


def drr_error(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
              direct_window_ms: float = 2.5) -> Dict[str, float]:
    """
    Compute DRR error between generated and reference RIR.
    
    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        sr: Sample rate
        direct_window_ms: Window for direct sound
    
    Returns:
        Dictionary with DRR values and error
    """
    drr_gen = compute_drr(rir_gen, sr, direct_window_ms)
    drr_ref = compute_drr(rir_ref, sr, direct_window_ms)
    
    error = drr_gen - drr_ref
    if np.isinf(drr_gen) or np.isinf(drr_ref):
        error = np.nan
    
    return {
        'error': error,
        'generated': drr_gen,
        'reference': drr_ref
    }


# =============================================================================
# Clarity (C50/C80)
# =============================================================================

def compute_clarity(rir: np.ndarray, sr: int, time_ms: float = 50) -> float:
    """
    Compute Clarity index (C50 or C80).
    
    C_t = 10 * log10(E_early / E_late)
    
    Where early is [0, t] and late is [t, inf]
    C50: t=50ms (speech), C80: t=80ms (music)
    
    Args:
        rir: Room impulse response
        sr: Sample rate
        time_ms: Cutoff time in milliseconds (50 for C50, 80 for C80)
    
    Returns:
        Clarity in dB
    """
    rir = np.asarray(rir).flatten()
    
    # Find direct sound as reference point
    direct_idx = find_direct_sound_idx(rir)
    
    # Cutoff sample (relative to direct sound)
    cutoff_samples = int(time_ms * sr / 1000)
    cutoff_idx = direct_idx + cutoff_samples
    
    if cutoff_idx >= len(rir):
        return np.inf
    
    # Early and late energy
    e_early = np.sum(rir[direct_idx:cutoff_idx] ** 2)
    e_late = np.sum(rir[cutoff_idx:] ** 2)
    
    if e_late < np.finfo(float).eps:
        return np.inf
    
    clarity = 10 * np.log10(e_early / e_late)
    
    return clarity


def compute_c50(rir: np.ndarray, sr: int) -> float:
    """Compute C50 (Clarity for speech, 50ms cutoff)."""
    return compute_clarity(rir, sr, time_ms=50)


def compute_c80(rir: np.ndarray, sr: int) -> float:
    """Compute C80 (Clarity for music, 80ms cutoff)."""
    return compute_clarity(rir, sr, time_ms=80)


def clarity_error(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
                  time_ms: float = 50) -> Dict[str, float]:
    """Compute clarity error between generated and reference RIR."""
    c_gen = compute_clarity(rir_gen, sr, time_ms)
    c_ref = compute_clarity(rir_ref, sr, time_ms)
    
    error = c_gen - c_ref
    if np.isinf(c_gen) or np.isinf(c_ref):
        error = np.nan
    
    return {
        'error': error,
        'generated': c_gen,
        'reference': c_ref
    }


# =============================================================================
# LSD (Log-Spectral Distance)
# =============================================================================

def compute_lsd(rir_gen: np.ndarray, rir_ref: np.ndarray, 
                n_fft: int = 2048, eps: float = 1e-10) -> float:
    """
    Compute Log-Spectral Distance (LSD).
    
    LSD = sqrt(mean((10*log10(|H_ref|²) - 10*log10(|H_gen|²))²))
    
    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        n_fft: FFT size
        eps: Small constant for numerical stability
    
    Returns:
        LSD in dB
    """
    rir_gen = np.asarray(rir_gen).flatten()
    rir_ref = np.asarray(rir_ref).flatten()
    
    # Zero-pad to same length
    max_len = max(len(rir_gen), len(rir_ref), n_fft)
    rir_gen_pad = np.zeros(max_len)
    rir_ref_pad = np.zeros(max_len)
    rir_gen_pad[:len(rir_gen)] = rir_gen
    rir_ref_pad[:len(rir_ref)] = rir_ref
    
    # Compute magnitude spectra
    H_gen = np.abs(np.fft.rfft(rir_gen_pad, n=n_fft))
    H_ref = np.abs(np.fft.rfft(rir_ref_pad, n=n_fft))
    
    # Log power spectra
    log_H_gen = 10 * np.log10(H_gen ** 2 + eps)
    log_H_ref = 10 * np.log10(H_ref ** 2 + eps)
    
    # LSD
    lsd = np.sqrt(np.mean((log_H_ref - log_H_gen) ** 2))
    
    return lsd


def compute_lsd_octave_bands(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
                              center_freqs: List[float] = None) -> Dict[float, float]:
    """
    Compute LSD per octave band.
    
    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        sr: Sample rate
        center_freqs: Octave band center frequencies
    
    Returns:
        Dictionary mapping center frequency to LSD
    """
    if center_freqs is None:
        center_freqs = [125, 250, 500, 1000, 2000, 4000]
    
    rir_gen = np.asarray(rir_gen).flatten()
    rir_ref = np.asarray(rir_ref).flatten()
    
    lsds = {}
    nyquist = sr / 2
    
    for fc in center_freqs:
        f_low = fc / np.sqrt(2)
        f_high = fc * np.sqrt(2)
        
        if f_low >= nyquist:
            continue
        f_high = min(f_high, nyquist * 0.99)
        
        try:
            sos = signal.butter(4, [f_low, f_high], btype='band', fs=sr, output='sos')
            gen_filt = signal.sosfilt(sos, rir_gen)
            ref_filt = signal.sosfilt(sos, rir_ref)
            lsds[fc] = compute_lsd(gen_filt, ref_filt)
        except Exception:
            continue
    
    return lsds


# =============================================================================
# Cosine Similarity
# =============================================================================

def compute_cosine_similarity(rir_gen: Union[np.ndarray, List[np.ndarray]],
                               rir_ref: Union[np.ndarray, List[np.ndarray]]) -> Union[float, np.ndarray]:
    """
    Compute cosine similarity between RIRs (supports single pair or batch).

    Cosine similarity measures the angular similarity between two vectors,
    ranging from -1 (opposite) to 1 (identical direction).

    cos_sim = (gen · ref) / (||gen|| * ||ref||)

    Assumes all RIRs have the same length (already aligned/truncated).

    Args:
        rir_gen: Generated RIR(s) - single 1D array or list of 1D arrays
        rir_ref: Reference RIR(s) - single 1D array or list of 1D arrays

    Returns:
        Cosine similarity (higher is better, 1.0 is perfect match)
        - If inputs are single arrays: returns float
        - If inputs are lists: returns np.ndarray of shape (batch_size,)
    """
    # Convert single arrays to list for uniform processing
    is_single = isinstance(rir_gen, np.ndarray) and rir_gen.ndim == 1
    if is_single:
        rir_gen = [rir_gen]
        rir_ref = [rir_ref]

    # Stack into matrix [batch_size, rir_length]
    gen_matrix = np.stack([np.asarray(g).flatten() for g in rir_gen], axis=0)
    ref_matrix = np.stack([np.asarray(r).flatten() for r in rir_ref], axis=0)

    # Vectorized computation: dot product along last axis
    dot_products = np.sum(gen_matrix * ref_matrix, axis=1)

    # Vectorized norms
    norms_gen = np.linalg.norm(gen_matrix, axis=1)
    norms_ref = np.linalg.norm(ref_matrix, axis=1)

    # Cosine similarity
    denominators = norms_gen * norms_ref
    cosine_sims = np.where(denominators > 0, dot_products / denominators, 0.0)

    # Return scalar if input was single pair
    return float(cosine_sims[0]) if is_single else cosine_sims


# =============================================================================
# EDC Distance
# =============================================================================

def edc_distance(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
                 metric: str = 'mse', db_range: float = -60) -> float:
    """
    Compute distance between EDCs.
    
    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        sr: Sample rate
        metric: 'mse', 'mae', or 'rmse'
        db_range: Only compare down to this dB level
    
    Returns:
        EDC distance
    """
    edc_gen = compute_edc(rir_gen)
    edc_ref = compute_edc(rir_ref)
    
    # Align lengths
    min_len = min(len(edc_gen), len(edc_ref))
    edc_gen = edc_gen[:min_len]
    edc_ref = edc_ref[:min_len]
    
    # Only compare above db_range
    valid_idx = edc_ref > db_range
    if not np.any(valid_idx):
        valid_idx = np.ones(len(edc_ref), dtype=bool)
    
    edc_gen = edc_gen[valid_idx]
    edc_ref = edc_ref[valid_idx]
    
    diff = edc_gen - edc_ref
    
    if metric == 'mse':
        return np.mean(diff ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(diff))
    elif metric == 'rmse':
        return np.sqrt(np.mean(diff ** 2))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def edc_distance_octave_bands(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
                               center_freqs: List[float] = None,
                               metric: str = 'mse') -> Dict[float, float]:
    """Compute EDC distance per octave band."""
    if center_freqs is None:
        center_freqs = [125, 250, 500, 1000, 2000, 4000]
    
    rir_gen = np.asarray(rir_gen).flatten()
    rir_ref = np.asarray(rir_ref).flatten()
    
    distances = {}
    nyquist = sr / 2
    
    for fc in center_freqs:
        f_low = fc / np.sqrt(2)
        f_high = fc * np.sqrt(2)
        
        if f_low >= nyquist:
            continue
        f_high = min(f_high, nyquist * 0.99)
        
        try:
            sos = signal.butter(4, [f_low, f_high], btype='band', fs=sr, output='sos')
            gen_filt = signal.sosfilt(sos, rir_gen)
            ref_filt = signal.sosfilt(sos, rir_ref)
            distances[fc] = edc_distance(gen_filt, ref_filt, sr, metric)
        except Exception:
            continue
    
    return distances


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def evaluate_rir_pair(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
                      center_freqs: List[float] = None) -> Dict:
    """
    Comprehensive evaluation of a generated RIR against reference.
    
    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        sr: Sample rate
        center_freqs: Octave band center frequencies
    
    Returns:
        Dictionary containing all metrics
    """
    if center_freqs is None:
        center_freqs = [125, 250, 500, 1000, 2000, 4000]
    
    results = {
        # T60
        't60': t60_error(rir_gen, rir_ref, sr, center_freqs),

        # EDT
        'edt': edt_error(rir_gen, rir_ref, sr),

        # DRR
        'drr': drr_error(rir_gen, rir_ref, sr),

        # Clarity
        'c50': clarity_error(rir_gen, rir_ref, sr, time_ms=50),
        'c80': clarity_error(rir_gen, rir_ref, sr, time_ms=80),

        # Spectral
        'lsd': {
            'broadband': compute_lsd(rir_gen, rir_ref),
            'per_band': compute_lsd_octave_bands(rir_gen, rir_ref, sr, center_freqs)
        },

        # EDC
        'edc_distance': {
            'broadband': edc_distance(rir_gen, rir_ref, sr),
            'per_band': edc_distance_octave_bands(rir_gen, rir_ref, sr, center_freqs)
        },

        # Cosine Similarity
        'cosine_similarity': compute_cosine_similarity(rir_gen, rir_ref)
    }

    return results


def evaluate_rir_batch(rirs_gen: List[np.ndarray], rirs_ref: List[np.ndarray], 
                       sr: int, center_freqs: List[float] = None) -> Dict:
    """
    Evaluate a batch of generated RIRs against references.
    
    Args:
        rirs_gen: List of generated RIRs
        rirs_ref: List of reference RIRs
        sr: Sample rate
        center_freqs: Octave band center frequencies
    
    Returns:
        Dictionary with individual and aggregate metrics
    """
    assert len(rirs_gen) == len(rirs_ref), "Number of generated and reference RIRs must match"
    
    individual_results = []
    for rir_gen, rir_ref in zip(rirs_gen, rirs_ref):
        individual_results.append(evaluate_rir_pair(rir_gen, rir_ref, sr, center_freqs))
    
    # Aggregate results
    aggregate = aggregate_metrics(individual_results)
    
    return {
        'individual': individual_results,
        'aggregate': aggregate,
        'n_samples': len(rirs_gen)
    }


def aggregate_metrics(results: List[Dict]) -> Dict:
    """
    Aggregate individual metrics into summary statistics.
    
    Args:
        results: List of individual evaluation results
    
    Returns:
        Dictionary with mean, std, median for each metric
    """
    def safe_stats(values):
        values = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]
        if not values:
            return {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'n_valid': 0}
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'n_valid': len(values)
        }
    
    aggregate = {}
    
    # T60
    t60_errors = [r['t60']['broadband'] for r in results]
    t60_abs_errors = [abs(e) for e in t60_errors if e is not None and not np.isnan(e)]
    aggregate['t60_error'] = safe_stats(t60_errors)
    aggregate['t60_abs_error'] = safe_stats(t60_abs_errors)
    aggregate['t60_mean_band_abs_error'] = safe_stats([r['t60']['mean_band_abs_error'] for r in results])
    
    # EDT
    aggregate['edt_error'] = safe_stats([r['edt']['error'] for r in results])
    
    # DRR
    aggregate['drr_error'] = safe_stats([r['drr']['error'] for r in results])
    aggregate['drr_abs_error'] = safe_stats([abs(r['drr']['error']) for r in results 
                                              if r['drr']['error'] is not None and not np.isnan(r['drr']['error'])])
    
    # Clarity
    aggregate['c50_error'] = safe_stats([r['c50']['error'] for r in results])
    aggregate['c80_error'] = safe_stats([r['c80']['error'] for r in results])
    
    # LSD
    aggregate['lsd'] = safe_stats([r['lsd']['broadband'] for r in results])

    # EDC distance
    aggregate['edc_distance'] = safe_stats([r['edc_distance']['broadband'] for r in results])

    # Cosine Similarity
    aggregate['cosine_similarity'] = safe_stats([r['cosine_similarity'] for r in results])

    return aggregate


# =============================================================================
# Utility Functions
# =============================================================================

def align_rir_lengths(rir1: np.ndarray, rir2: np.ndarray, 
                      mode: str = 'truncate') -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two RIRs to the same length.
    
    Args:
        rir1, rir2: Input RIRs
        mode: 'truncate' (use shorter) or 'pad' (zero-pad shorter)
    
    Returns:
        Aligned RIRs
    """
    rir1 = np.asarray(rir1).flatten()
    rir2 = np.asarray(rir2).flatten()
    
    if mode == 'truncate':
        min_len = min(len(rir1), len(rir2))
        return rir1[:min_len], rir2[:min_len]
    elif mode == 'pad':
        max_len = max(len(rir1), len(rir2))
        rir1_pad = np.zeros(max_len)
        rir2_pad = np.zeros(max_len)
        rir1_pad[:len(rir1)] = rir1
        rir2_pad[:len(rir2)] = rir2
        return rir1_pad, rir2_pad
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Quick test with synthetic RIRs
    sr = 16000
    t = np.arange(sr) / sr  # 1 second
    
    # Synthetic RIR: exponential decay with some noise
    rir_ref = np.exp(-5 * t) * np.random.randn(sr) * 0.1
    rir_ref[100] = 1.0  # Direct sound
    
    # Slightly different generated RIR
    rir_gen = np.exp(-4.5 * t) * np.random.randn(sr) * 0.1
    rir_gen[105] = 0.95
    
    print("Testing acoustic metrics...")
    results = evaluate_rir_pair(rir_gen, rir_ref, sr)
    
    print(f"\nT60 (ref): {results['t60']['broadband_ref']:.3f}s")
    print(f"T60 (gen): {results['t60']['broadband_gen']:.3f}s")
    print(f"T60 error: {results['t60']['broadband']:.3f}s")
    print(f"\nDRR (ref): {results['drr']['reference']:.2f} dB")
    print(f"DRR (gen): {results['drr']['generated']:.2f} dB")
    print(f"\nLSD: {results['lsd']['broadband']:.2f} dB")
    print(f"EDC distance (MSE): {results['edc_distance']['broadband']:.2f}")
    
    print("\n✓ All metrics computed successfully!")