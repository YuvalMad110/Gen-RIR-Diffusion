"""
Scoring one generated RIR against its reference, and aggregating a split's worth of scores.

The rules this module applies, as opposed to the measurements it calls:

  * Neither signal is aligned in length or in level by default. Each is measured at its own
    length, so a generated RIR is never flattered by being cut to fit its reference.
  * T60, C50 and DRR for the reference come from the store, which caches what the preparation
    scripts measured with these same functions on the same native-length signal. Only the
    quantities with no cached counterpart are measured here.
  * LSD needs one length for both signals, so it is the exception. It runs at three lengths,
    each of which answers a different question, and each plain and with the peaks aligned.

utils/acoustic_metrics.py holds the measurements themselves and is shared with the preparation
scripts and the baselines; nothing here changes what it reports.
"""

import numpy as np

from utils.acoustic_metrics import (DEFAULT_OCTAVE_BANDS, DEFAULT_T60_FIT_RANGE, align_peaks,
                                    compute_c50, compute_clarity, compute_cosine_similarity,
                                    compute_drr, compute_edt, compute_lsd, compute_t60,
                                    compute_t60_crossings, compute_t60_octave_bands,
                                    edc_distance, peak_shift)

# L is the length both signals are brought to before an LSD. Each answers a different question:
#   gen  everything the model produced, so a tail the reference does not have is charged for
#   ref  the whole reference, so a tail the model failed to produce is charged for
#   min  the overlap only, charging for neither
LSD_LENGTHS = ('gen', 'ref', 'min')


# =============================================================================
# Length handling, for LSD alone
# =============================================================================

def lsd_length(generated, reference, variant: str) -> int:
    """The length an LSD variant compares at."""
    if variant == 'gen':
        return len(generated)
    if variant == 'ref':
        return len(reference)
    if variant == 'min':
        return min(len(generated), len(reference))
    raise ValueError(f'unknown LSD length variant {variant!r}, expected one of {LSD_LENGTHS}')


def at_length(signal_in, length: int) -> np.ndarray:
    """One signal brought to `length`: zero-padded when short, truncated when long."""
    x = np.asarray(signal_in, dtype=np.float64).flatten()
    return np.pad(x, (0, length - len(x))) if len(x) < length else x[:length]


def lsd_at_length(generated, reference, sr: int, variant: str, dry_signal=None,
                  align: bool = False) -> float:
    """
    LSD with both signals brought to one length first.

    Matching happens before any speech convolution, so the convolved pair is the same length by
    construction and no silent re-padding can occur inside the distance.

    Args:
        generated, reference: the RIRs
        sr:                   sample rate
        variant:              one of LSD_LENGTHS
        dry_signal:           speech to convolve with both, or None to compare the RIRs directly
        align:                shift the generated so its peak sits on the reference's first

    Returns:
        LSD in dB.
    """
    length = lsd_length(generated, reference, variant)
    gen, ref = at_length(generated, length), at_length(reference, length)
    if align:
        gen, ref = align_peaks(gen, ref)
    if len(gen) != len(ref):
        raise ValueError(f'LSD needs equal lengths, got {len(gen)} and {len(ref)}')

    return compute_lsd(gen, ref, sr, dry_signal=dry_signal)


# =============================================================================
# Scoring one pair
# =============================================================================

def _signed_errors(prefix: str, gen: float, ref: float) -> dict:
    """A measured pair as its two values and the error between them, as gen - ref."""
    error = gen - ref if np.isfinite(gen) and np.isfinite(ref) else np.nan
    out = {f'{prefix}_gen': gen, f'{prefix}_ref': ref, f'{prefix}_err': error,
           f'{prefix}_abs': abs(error) if np.isfinite(error) else np.nan}
    if np.isfinite(error) and ref != 0:
        out[f'{prefix}_perc'] = abs(error) / abs(ref) * 100
    else:
        out[f'{prefix}_perc'] = np.nan
    return out


def score_pair(generated, reference, reference_metrics: dict, sr: int, dry_signal=None,
               t60_fit_range=DEFAULT_T60_FIT_RANGE, octave_bands=DEFAULT_OCTAVE_BANDS) -> dict:
    """
    Every metric for one generated RIR against its reference, as a flat dict of scalars.

    Args:
        generated:         the generated RIR, at whatever length the model produces
        reference:         the reference RIR, at its native length
        reference_metrics: the store's cached values for this pair, with keys 'rt60', 'c50', 'drr'
        sr:                sample rate
        dry_signal:        speech to convolve for the LSD, or None to compare the RIRs directly
        t60_fit_range:     dB window compute_t60 fits over
        octave_bands:      centre frequencies for the per-band T60

    Returns:
        A flat dict. Errors are gen - ref throughout; '_abs' and '_perc' are magnitudes.
    """
    gen = np.asarray(generated, dtype=np.float64).flatten()
    ref = np.asarray(reference, dtype=np.float64).flatten()
    row = {}

    # T60 by the fitted window, the reference value taken from the store
    row.update(_signed_errors('t60', compute_t60(gen, sr, fit_range=t60_fit_range),
                              float(reference_metrics['rt60'])))

    # T60 by the -5/-65 dB crossings, which has no cached counterpart
    row.update(_signed_errors('t60x', compute_t60_crossings(gen, sr), compute_t60_crossings(ref, sr)))

    # Clarity and direct-to-reverberant, both reference values from the store
    row.update(_signed_errors('c50', compute_c50(gen, sr), float(reference_metrics['c50'])))
    row.update(_signed_errors('drr', compute_drr(gen, sr), float(reference_metrics['drr'])))
    row.update(_signed_errors('c80', compute_clarity(gen, sr, 80), compute_clarity(ref, sr, 80)))
    row.update(_signed_errors('edt', compute_edt(gen, sr), compute_edt(ref, sr)))

    # Per band, by the fitted window only
    gen_bands = compute_t60_octave_bands(gen, sr, octave_bands, fit_range=t60_fit_range)
    ref_bands = compute_t60_octave_bands(ref, sr, octave_bands, fit_range=t60_fit_range)
    for fc in octave_bands:
        g, r = gen_bands.get(fc, np.nan), ref_bands.get(fc, np.nan)
        row[f't60_band_{int(fc)}_err'] = g - r if np.isfinite(g) and np.isfinite(r) else np.nan
        row[f't60_band_{int(fc)}_abs'] = abs(row[f't60_band_{int(fc)}_err'])

    # LSD at each length, plain and with the peaks aligned
    for variant in LSD_LENGTHS:
        row[f'lsd_{variant}'] = lsd_at_length(gen, ref, sr, variant, dry_signal, align=False)
        row[f'lsda_{variant}'] = lsd_at_length(gen, ref, sr, variant, dry_signal, align=True)

    # Where the direct sound landed
    shift = peak_shift(gen, ref)
    row['peak_shift_samples'] = shift
    row['peak_shift_ms'] = shift / sr * 1000

    # These two compare the overlap: edc_distance truncates to it itself, and a cosine
    # similarity over padding would read as agreement where there is no signal
    overlap = min(len(gen), len(ref))
    row['edc_distance'] = edc_distance(gen, ref, sr)
    row['cosine_similarity'] = float(compute_cosine_similarity(gen[:overlap], ref[:overlap]))

    row['gen_length'] = len(gen)
    row['ref_length'] = len(ref)
    return row


# =============================================================================
# Aggregating a split
# =============================================================================

def aggregate(rows: list) -> dict:
    """
    Mean, median, spread and valid count for every metric across a split.

    A metric is counted valid for the samples where it came out finite, so a measurement that
    fails on some pairs reduces its own count rather than poisoning the mean. Correlations are
    added for the quantities whose agreement across rooms is the point, not just their error.

    'rmse' is the root mean square of the metric's own values over the valid samples. Taken on
    an '_err' key that is the RMSE against the reference; taken on an unsigned metric such as
    an LSD it is the root mean square of the distances themselves.

    Returns:
        {metric: {'mean', 'median', 'std', 'rmse', 'n'}} plus 't60_corr' and 't60x_corr'.
    """
    if not rows:
        return {}

    out = {}
    for key in rows[0]:
        values = np.array([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        out[key] = {'mean': float(finite.mean()) if finite.size else np.nan,
                    'median': float(np.median(finite)) if finite.size else np.nan,
                    'std': float(finite.std()) if finite.size else np.nan,
                    'rmse': float(np.sqrt((finite ** 2).mean())) if finite.size else np.nan,
                    'n': int(finite.size)}

    for prefix in ('t60', 't60x'):
        gen = np.array([row.get(f'{prefix}_gen', np.nan) for row in rows], dtype=np.float64)
        ref = np.array([row.get(f'{prefix}_ref', np.nan) for row in rows], dtype=np.float64)
        both = np.isfinite(gen) & np.isfinite(ref)
        out[f'{prefix}_corr'] = (float(np.corrcoef(gen[both], ref[both])[0, 1])
                                 if both.sum() > 2 and ref[both].std() > 0 else np.nan)

    return out
