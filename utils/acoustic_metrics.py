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

DEFAULT_T60_FIT_RANGE: Tuple[float, float] = (-5, -25)
DEFAULT_OCTAVE_BANDS: List[float] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]


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
        center_freqs = DEFAULT_OCTAVE_BANDS
    
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
                          fit_range: Tuple[float, float] = DEFAULT_T60_FIT_RANGE) -> float:
    """
    Estimate T60 from EDC using linear regression.
    
    Fits a line over the `fit_range` dB window of the EDC and extrapolates to -60 dB.
    
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


def compute_t60(rir: np.ndarray, sr: int, fit_range: Tuple[float, float] = DEFAULT_T60_FIT_RANGE) -> float:
    """
    Compute broadband T60 from RIR.

    Args:
        rir: Room impulse response
        sr: Sample rate
        fit_range: (start_db, end_db) window for EDC slope estimation

    Returns:
        T60 in seconds
    """
    edc_db = compute_edc(rir)
    return estimate_t60_from_edc(edc_db, sr, fit_range)


def compute_t60_batch(rirs: List[np.ndarray], sr: int, fit_range: Tuple[float, float] = DEFAULT_T60_FIT_RANGE) -> List[float]:
    """Compute broadband T60 for a list of RIRs. Raises ValueError if any estimate returns NaN."""
    results = []
    for i, rir in enumerate(rirs):
        t60 = compute_t60(rir, sr, fit_range)
        if np.isnan(t60):
            raise ValueError(f"RT60 estimation returned NaN for RIR at index {i}.")
        results.append(t60)
    return results


def compute_t60_octave_bands(rir: np.ndarray, sr: int, center_freqs: List[float] = None,
                              fit_range: Tuple[float, float] = DEFAULT_T60_FIT_RANGE) -> Dict[float, float]:
    """
    Compute T60 for each octave band.

    Args:
        rir: Room impulse response
        sr: Sample rate
        center_freqs: Octave band center frequencies
        fit_range: (start_db, end_db) window for EDC slope estimation

    Returns:
        Dictionary mapping center frequency to T60 (seconds)
    """
    edcs = compute_edc_octave_bands(rir, sr, center_freqs)
    t60s = {}

    for fc, edc_db in edcs.items():
        t60s[fc] = estimate_t60_from_edc(edc_db, sr, fit_range)

    return t60s


def t60_error(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
              center_freqs: List[float] = None, fit_range: Tuple[float, float] = DEFAULT_T60_FIT_RANGE) -> Dict[str, Union[float, Dict[float, float]]]:
    """
    Compute T60 error between generated and reference RIR.

    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        sr: Sample rate
        center_freqs: Octave band center frequencies
        fit_range: (start_db, end_db) window for EDC slope estimation

    Returns:
        Dictionary with:
        - 'broadband': Absolute T60 error (seconds)
        - 'per_band': Dict of per-band T60 errors
        - 'mean_band_error': Mean absolute error across bands
    """
    # Broadband
    t60_gen = compute_t60(rir_gen, sr, fit_range)
    t60_ref = compute_t60(rir_ref, sr, fit_range)

    # Per octave band
    t60_gen_bands = compute_t60_octave_bands(rir_gen, sr, center_freqs, fit_range)
    t60_ref_bands = compute_t60_octave_bands(rir_ref, sr, center_freqs, fit_range)
    
    band_errors = {}
    band_perc_errors = {}
    for fc in t60_gen_bands:
        if fc in t60_ref_bands:
            gen_b = t60_gen_bands[fc]
            ref_b = t60_ref_bands[fc]
            if not np.isnan(gen_b) and not np.isnan(ref_b):
                band_errors[fc] = gen_b - ref_b
                if ref_b != 0:
                    band_perc_errors[fc] = abs(gen_b - ref_b) / ref_b * 100

    valid_errors = [e for e in band_errors.values() if not np.isnan(e)]
    mean_error = np.mean(np.abs(valid_errors)) if valid_errors else np.nan

    # Compute percentage error (guard only against division by zero)
    perc_error = abs(t60_gen - t60_ref) / t60_ref * 100 if t60_ref != 0 else np.nan

    return {
        'broadband': t60_gen - t60_ref if not (np.isnan(t60_gen) or np.isnan(t60_ref)) else np.nan,
        'broadband_gen': t60_gen,
        'broadband_ref': t60_ref,
        'perc': perc_error,
        'per_band': band_errors,
        'per_band_perc': band_perc_errors,
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

def compute_lsd(rir_gen: np.ndarray, rir_ref: np.ndarray, dry_signal: np.ndarray = None, n_fft: int = 2048, eps: float = 1e-10) -> float:
    """
    Compute Log-Spectral Distance (LSD).

    LSD = sqrt(mean((10*log10(|H_ref|²) - 10*log10(|H_gen|²))²))

    If dry_signal is provided, computes LSD on reverbed signals (RIR convolved with dry signal).
    Otherwise computes LSD directly on the RIR frequency responses.

    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        dry_signal: Optional dry signal to convolve with RIRs for reverbed LSD
        n_fft: FFT size
        eps: Small constant for numerical stability

    Returns:
        LSD in dB
    """
    rir_gen = np.asarray(rir_gen).flatten()
    rir_ref = np.asarray(rir_ref).flatten()

    # If dry signal provided, compute reverbed signals
    if dry_signal is not None:
        dry_signal = np.asarray(dry_signal).flatten()
        sig_gen = signal.fftconvolve(dry_signal, rir_gen, mode='full')
        sig_ref = signal.fftconvolve(dry_signal, rir_ref, mode='full')
    else:
        sig_gen = rir_gen
        sig_ref = rir_ref

    # Zero-pad to same length
    max_len = max(len(sig_gen), len(sig_ref), n_fft)
    sig_gen_pad = np.zeros(max_len)
    sig_ref_pad = np.zeros(max_len)
    sig_gen_pad[:len(sig_gen)] = sig_gen
    sig_ref_pad[:len(sig_ref)] = sig_ref

    # Compute magnitude spectra
    H_gen = np.abs(np.fft.rfft(sig_gen_pad, n=n_fft))
    H_ref = np.abs(np.fft.rfft(sig_ref_pad, n=n_fft))

    # Log power spectra
    log_H_gen = 10 * np.log10(H_gen ** 2 + eps)
    log_H_ref = 10 * np.log10(H_ref ** 2 + eps)

    # LSD
    lsd = np.sqrt(np.mean((log_H_ref - log_H_gen) ** 2))

    return lsd


def compute_lsd_octave_bands(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int, center_freqs: List[float] = None, dry_signal: np.ndarray = None) -> Dict[float, float]:
    """
    Compute LSD per octave band.

    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        sr: Sample rate
        center_freqs: Octave band center frequencies
        dry_signal: Optional dry signal to convolve with RIRs for reverbed LSD

    Returns:
        Dictionary mapping center frequency to LSD
    """
    if center_freqs is None:
        center_freqs = DEFAULT_OCTAVE_BANDS

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
            lsds[fc] = compute_lsd(gen_filt, ref_filt, dry_signal=dry_signal)
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
        center_freqs = DEFAULT_OCTAVE_BANDS
    
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

ALL_METRICS = ['t60', 'edt', 'drr', 'c50', 'c80', 'lsd', 'edc_distance', 'cosine_similarity']


def evaluate_rir_pair(rir_gen: np.ndarray, rir_ref: np.ndarray, sr: int,
                      center_freqs: List[float] = None, dry_signal: np.ndarray = None,
                      metrics: List[str] = None,
                      fit_range: Tuple[float, float] = DEFAULT_T60_FIT_RANGE) -> Dict:
    """
    Comprehensive evaluation of a generated RIR against reference.

    Args:
        rir_gen: Generated RIR
        rir_ref: Reference RIR
        sr: Sample rate
        center_freqs: Octave band center frequencies
        dry_signal: Optional dry signal to convolve with RIRs for reverbed LSD
        metrics: List of metric keys to compute (default: all). Options: t60, edt, drr, c50, c80, lsd, edc_distance, cosine_similarity
        fit_range: (start_db, end_db) window for T60 EDC slope estimation

    Returns:
        Dictionary containing requested metrics
    """
    if center_freqs is None:
        center_freqs = DEFAULT_OCTAVE_BANDS
    if metrics is None:
        metrics = ALL_METRICS

    results = {}
    if 't60' in metrics:
        results['t60'] = t60_error(rir_gen, rir_ref, sr, center_freqs, fit_range)
    if 'edt' in metrics:
        results['edt'] = edt_error(rir_gen, rir_ref, sr)
    if 'drr' in metrics:
        results['drr'] = drr_error(rir_gen, rir_ref, sr)
    if 'c50' in metrics:
        results['c50'] = clarity_error(rir_gen, rir_ref, sr, time_ms=50)
    if 'c80' in metrics:
        results['c80'] = clarity_error(rir_gen, rir_ref, sr, time_ms=80)
    if 'lsd' in metrics:
        results['lsd'] = {
            'broadband': compute_lsd(rir_gen, rir_ref, dry_signal=dry_signal),
            'per_band': compute_lsd_octave_bands(rir_gen, rir_ref, sr, center_freqs, dry_signal=dry_signal)
        }
    if 'edc_distance' in metrics:
        results['edc_distance'] = {
            'broadband': edc_distance(rir_gen, rir_ref, sr),
            'per_band': edc_distance_octave_bands(rir_gen, rir_ref, sr, center_freqs)
        }
    if 'cosine_similarity' in metrics:
        results['cosine_similarity'] = compute_cosine_similarity(rir_gen, rir_ref)

    return results


def evaluate_rir_batch(rirs_gen: List[np.ndarray], rirs_ref: List[np.ndarray],
                       sr: int, center_freqs: List[float] = None, fit_range: Tuple[float, float] = DEFAULT_T60_FIT_RANGE) -> Dict:
    """
    Evaluate a batch of generated RIRs against references.

    Args:
        rirs_gen: List of generated RIRs
        rirs_ref: List of reference RIRs
        sr: Sample rate
        center_freqs: Octave band center frequencies
        fit_range: (start_db, end_db) window for T60 EDC slope estimation

    Returns:
        Dictionary with individual and aggregate metrics
    """
    assert len(rirs_gen) == len(rirs_ref), "Number of generated and reference RIRs must match"

    individual_results = []
    for rir_gen, rir_ref in zip(rirs_gen, rirs_ref):
        individual_results.append(evaluate_rir_pair(rir_gen, rir_ref, sr, center_freqs, fit_range=fit_range))
    
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
    has = lambda key: results and key in results[0]

    if has('t60'):
        t60_errors = [r['t60']['broadband'] for r in results]
        t60_abs_errors = [abs(e) for e in t60_errors if e is not None and not np.isnan(e)]
        aggregate['t60_error'] = safe_stats(t60_errors)
        aggregate['t60_abs_error'] = safe_stats(t60_abs_errors)
        aggregate['t60_perc_error'] = safe_stats([r['t60']['perc'] for r in results])
        aggregate['t60_mean_band_abs_error'] = safe_stats([r['t60']['mean_band_abs_error'] for r in results])
        all_bands = sorted({fc for r in results for fc in r['t60'].get('per_band_perc', {})})
        aggregate['t60_band_perc_error'] = {
            fc: safe_stats([r['t60']['per_band_perc'][fc]
                            for r in results if fc in r['t60'].get('per_band_perc', {})])
            for fc in all_bands
        }

    if has('edt'):
        aggregate['edt_error'] = safe_stats([r['edt']['error'] for r in results])

    if has('drr'):
        aggregate['drr_error'] = safe_stats([r['drr']['error'] for r in results])
        aggregate['drr_abs_error'] = safe_stats([abs(r['drr']['error']) for r in results
                                                  if r['drr']['error'] is not None and not np.isnan(r['drr']['error'])])

    if has('c50'):
        c50_errors = [r['c50']['error'] for r in results]
        aggregate['c50_error'] = safe_stats(c50_errors)
        aggregate['c50_abs_error'] = safe_stats([abs(e) for e in c50_errors if e is not None and not np.isnan(e)])
    if has('c80'):
        c80_errors = [r['c80']['error'] for r in results]
        aggregate['c80_error'] = safe_stats(c80_errors)
        aggregate['c80_abs_error'] = safe_stats([abs(e) for e in c80_errors if e is not None and not np.isnan(e)])

    if has('lsd'):
        aggregate['lsd'] = safe_stats([r['lsd']['broadband'] for r in results])

    if has('edc_distance'):
        aggregate['edc_distance'] = safe_stats([r['edc_distance']['broadband'] for r in results])

    if has('cosine_similarity'):
        aggregate['cosine_similarity'] = safe_stats([r['cosine_similarity'] for r in results])

    return aggregate


# =============================================================================
# Artifact Detection
# =============================================================================

def has_secondary_peak_artifact(rir: np.ndarray, sr: int,
                                 threshold_db: float = 14.0,
                                 reference_window_ms: float = 50.0,
                                 safety_gap_ms: float = 20.0,
                                 smoothing_ms: float = 20.0,
                                 min_delay_ms: float = 5.0,
                                 tail_ignore_ms: float = 100.0,
                                 save_path: Optional[str] = None) -> bool:
    """
    Detect if an RIR contains a secondary peak artifact that violates
    the expected monotonic decay of energy.

    Uses a state-based prominence approach: compares current energy at time t
    to the mean energy of a local past reference window. An artifact is detected
    when the current energy rises significantly above the recent past.

    Args:
        rir: Room impulse response (1D array)
        sr: Sampling frequency in Hz
        threshold_db: Required rise above reference mean to trigger detection (default: 25.0)
        reference_window_ms: Size of past window to average for reference (default: 50.0)
        safety_gap_ms: Gap between end of reference window and current point (default: 20.0)
        smoothing_ms: Window size in ms for Savitzky-Golay smoothing (default: 20.0)
        min_delay_ms: Minimum delay after main peak before checking (default: 5.0)
        tail_ignore_ms: Ignore the last N ms of the signal (default: 150.0)
        save_path: Optional path to save diagnostic plot. If None, no plot is saved.

    Returns:
        True if secondary peak artifact detected, False otherwise.
    """
    rir = np.asarray(rir).flatten()

    # 1. Compute analytic envelope using Hilbert Transform
    analytic_signal = signal.hilbert(rir)
    envelope = np.abs(analytic_signal)

    # 2. Convert to dB, normalized to 0 dB at maximum
    eps = 1e-10
    env_db = 20 * np.log10(envelope + eps)
    env_db -= np.max(env_db)

    # 3. Smooth the envelope with Savitzky-Golay filter
    savgol_window = int((smoothing_ms / 1000.0) * sr)
    savgol_window = min(savgol_window, len(env_db) // 2)
    if savgol_window % 2 == 0:
        savgol_window += 1
    savgol_window = max(savgol_window, 5)

    smoothed_env = signal.savgol_filter(env_db, savgol_window, polyorder=2)

    # 4. Find main peak and compute analysis start
    main_peak_idx = np.argmax(smoothed_env)
    min_delay_samples = int((min_delay_ms / 1000.0) * sr)

    # Convert parameters to samples
    reference_window_samples = int((reference_window_ms / 1000.0) * sr)
    safety_gap_samples = int((safety_gap_ms / 1000.0) * sr)
    tail_ignore_samples = int((tail_ignore_ms / 1000.0) * sr)

    # Analysis can start only after: main_peak + min_delay + reference_window + safety_gap
    min_valid_idx = main_peak_idx + min_delay_samples + reference_window_samples + safety_gap_samples
    start_idx = main_peak_idx + min_delay_samples  # For plotting purposes

    # Analysis stops before the last tail_ignore_ms
    max_valid_idx = len(smoothed_env) - tail_ignore_samples

    if min_valid_idx >= max_valid_idx:
        if save_path:
            _plot_secondary_peak_detection(
                rir, envelope, env_db, smoothed_env, sr,
                main_peak_idx, start_idx, np.array([]), np.array([]),
                threshold_db, reference_window_ms, safety_gap_ms, tail_ignore_ms,
                None, False, save_path
            )
        return False  # RIR too short to analyze

    # 5. Compute reference mean using cumulative sum for efficiency
    # reference_mean[t] = mean of smoothed_env from (t - safety_gap - ref_window) to (t - safety_gap)
    cumsum = np.cumsum(smoothed_env)

    # Vectorized computation of reference means
    # For index t, reference window is [t - safety_gap - ref_window, t - safety_gap)
    analysis_indices = np.arange(min_valid_idx, max_valid_idx)

    ref_end_indices = analysis_indices - safety_gap_samples  # End of reference window (exclusive)
    ref_start_indices = ref_end_indices - reference_window_samples  # Start of reference window

    # Compute cumsum differences for window means
    # cumsum[end-1] - cumsum[start-1] gives sum from start to end-1
    cumsum_at_end = cumsum[ref_end_indices - 1]
    cumsum_at_start = np.where(ref_start_indices > 0, cumsum[ref_start_indices - 1], 0)
    reference_means = (cumsum_at_end - cumsum_at_start) / reference_window_samples

    # Current envelope values at analysis points
    current_values = smoothed_env[analysis_indices]

    # Delta from local mean
    delta_from_mean = current_values - reference_means

    # 6. Detection: find where delta exceeds threshold
    artifact_detected = False
    artifact_idx = None

    above_threshold_mask = delta_from_mean > threshold_db
    if np.any(above_threshold_mask):
        artifact_detected = True
        first_artifact_pos = np.argmax(above_threshold_mask)
        artifact_idx = analysis_indices[first_artifact_pos]

    # Save diagnostic plot if requested
    if save_path:
        _plot_secondary_peak_detection(
            rir, envelope, env_db, smoothed_env, sr,
            main_peak_idx, start_idx, analysis_indices, delta_from_mean,
            threshold_db, reference_window_ms, safety_gap_ms, tail_ignore_ms,
            artifact_idx, artifact_detected, save_path
        )

    return artifact_detected


def _plot_secondary_peak_detection(rir, envelope, env_db, smoothed_env, sr,
                                    main_peak_idx, start_idx, analysis_indices,
                                    delta_from_mean, threshold_db,
                                    reference_window_ms, safety_gap_ms, tail_ignore_ms,
                                    artifact_idx, artifact_detected, save_path):
    """Helper function to create diagnostic plot for secondary peak detection."""
    import matplotlib.pyplot as plt

    time_ms = np.arange(len(rir)) / sr * 1000  # Time in milliseconds

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Colors
    color_waveform = '#1f77b4'
    color_envelope = '#ff7f0e'
    color_smoothed = '#2ca02c'
    color_threshold = '#d62728'
    color_artifact = '#d62728'
    color_reference = '#9467bd'

    # --- Plot 1: Waveform + Envelope ---
    ax1 = axes[0]
    ax1.plot(time_ms, rir, color=color_waveform, alpha=0.6, linewidth=0.5, label='Waveform')
    ax1.plot(time_ms, envelope, color=color_envelope, linewidth=1.5, label='Hilbert Envelope')
    ax1.axvline(x=main_peak_idx / sr * 1000, color='green', linestyle='--',
                linewidth=1.5, label=f'Main Peak ({main_peak_idx / sr * 1000:.1f} ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Waveform with Hilbert Envelope')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_ms[-1])

    # --- Plot 2: Smoothed Envelope (dB) with reference window illustration ---
    ax2 = axes[1]
    ax2.plot(time_ms, env_db, color=color_envelope, alpha=0.4, linewidth=0.8, label='Envelope (dB)')
    ax2.plot(time_ms, smoothed_env, color=color_smoothed, linewidth=1.5, label='Smoothed Envelope (dB)')
    ax2.axvline(x=main_peak_idx / sr * 1000, color='green', linestyle='--',
                linewidth=1.5, label='Main Peak')
    if start_idx < len(smoothed_env):
        ax2.axvline(x=start_idx / sr * 1000, color='purple', linestyle=':',
                    linewidth=1.5, label=f'Analysis Start ({start_idx / sr * 1000:.1f} ms)')
    if artifact_idx is not None:
        artifact_time_ms = artifact_idx / sr * 1000
        ax2.axvline(x=artifact_time_ms, color=color_artifact, linestyle='-',
                    linewidth=2, label=f'Artifact Detected ({artifact_time_ms:.1f} ms)')
        ax2.plot(artifact_time_ms, smoothed_env[artifact_idx], 'o',
                 color=color_artifact, markersize=10)

        # Draw reference window and gap illustration at artifact point
        ref_end_ms = artifact_time_ms - safety_gap_ms
        ref_start_ms = ref_end_ms - reference_window_ms
        y_level = smoothed_env[artifact_idx]

        # Reference window bracket
        ax2.annotate('', xy=(ref_start_ms, y_level - 5), xytext=(ref_end_ms, y_level - 5),
                     arrowprops=dict(arrowstyle='<->', color=color_reference, lw=2))
        ax2.text((ref_start_ms + ref_end_ms) / 2, y_level - 8, 'Ref Window',
                 ha='center', va='top', fontsize=9, color=color_reference)

        # Safety gap bracket
        ax2.annotate('', xy=(ref_end_ms, y_level - 5), xytext=(artifact_time_ms, y_level - 5),
                     arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
        ax2.text((ref_end_ms + artifact_time_ms) / 2, y_level - 8, 'Gap',
                 ha='center', va='top', fontsize=9, color='orange')

    # Show tail ignore region
    tail_start_ms = time_ms[-1] - tail_ignore_ms
    ax2.axvline(x=tail_start_ms, color='brown', linestyle='--',
                linewidth=1.5, label=f'Tail Ignore ({tail_start_ms:.0f} ms)')
    ax2.axvspan(tail_start_ms, time_ms[-1], alpha=0.1, color='brown')

    ax2.set_ylabel('Amplitude (dB)')
    ax2.set_title('Smoothed Envelope in dB')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-80)
    ax2.set_xlim(0, time_ms[-1])

    # --- Plot 3: Delta from Local Mean ---
    ax3 = axes[2]
    if len(delta_from_mean) > 0:
        analysis_times_ms = analysis_indices / sr * 1000
        ax3.plot(analysis_times_ms, delta_from_mean, color=color_smoothed,
                 linewidth=1.5, label='Delta from Local Mean')
        # Zero line
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.7, label='Zero (no change)')
        # Threshold line
        ax3.axhline(y=threshold_db, color=color_threshold, linestyle='--',
                    linewidth=2, label=f'Threshold (+{threshold_db} dB)')
        # Fill region above threshold
        ax3.fill_between(analysis_times_ms, delta_from_mean, threshold_db,
                         where=(delta_from_mean > threshold_db),
                         color=color_artifact, alpha=0.3)
        if artifact_idx is not None:
            artifact_pos = np.where(analysis_indices == artifact_idx)[0]
            if len(artifact_pos) > 0:
                ax3.plot(artifact_idx / sr * 1000, delta_from_mean[artifact_pos[0]], 'o',
                         color=color_artifact, markersize=10, label='Detection Point')

    # Add parameters text box
    params_text = (f'Parameters:\n'
                   f'  Reference Window: {reference_window_ms:.0f} ms\n'
                   f'  Safety Gap: {safety_gap_ms:.0f} ms\n'
                   f'  Threshold: {threshold_db:.1f} dB\n'
                   f'  Tail Ignore: {tail_ignore_ms:.0f} ms')
    ax3.text(0.02, 0.98, params_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax3.set_ylabel('Delta from Local Mean (dB)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title('Energy Rise Above Local Reference (current - mean of past window)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, time_ms[-1])

    # Overall title with detection result
    result_str = "ARTIFACT DETECTED" if artifact_detected else "No artifact detected"
    result_color = color_artifact if artifact_detected else 'green'
    fig.suptitle(f'Secondary Peak Artifact Detection: {result_str}',
                 fontsize=14, fontweight='bold', color=result_color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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