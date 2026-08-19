"""
Evaluation reporting utilities for RIR comparison.

Contains functions for saving evaluation summaries, metrics tables, and selected samples.
Used by full_model_eval.py and synthetic_eval.py.
"""

import numpy as np
import torch
from pathlib import Path

from utils.misc import get_israel_time


def format_metric_line(name, stats, unit):
    """Format a single metric line for the summary."""
    if stats['mean'] is None or np.isnan(stats['mean']):
        return f"  {name:20s}: N/A"
    return f"  {name:20s}: {stats['mean']:8.4f} ± {stats['std']:.4f} {unit}  (median: {stats['median']:.4f}, n_valid={stats['n_valid']})"


def _format_comparison_table(aggregate, baseline_aggregate, diff_all_metrics, baseline_all_metrics):
    """Format a three-column comparison table: Diffusion | Baseline | Baseline-Diff.

    Args:
        aggregate: Aggregated diffusion metrics
        baseline_aggregate: Aggregated baseline metrics
        diff_all_metrics: Per-sample diffusion metrics list
        baseline_all_metrics: Per-sample baseline metrics list

    Returns:
        List of formatted lines
    """
    lines = []

    # Define metrics to compare, with extractors for per-sample absolute values
    # For error metrics: lower = better. For cosine_similarity: higher = better.
    metrics_table = [
        ('T60 Abs Error',    't60_abs_error',   's',  lambda m: abs(m['t60']['broadband']) if m['t60']['broadband'] is not None and not np.isnan(m['t60']['broadband']) else np.nan),
        ('T60 Perc Error',   't60_perc_error',  '%',  lambda m: m['t60']['perc']),
        ('EDT Error',        'edt_error',       's',  lambda m: abs(m['edt']['error']) if m['edt']['error'] is not None and not np.isnan(m['edt']['error']) else np.nan),
        ('DRR Abs Error',    'drr_abs_error',   'dB', lambda m: abs(m['drr']['error']) if m['drr']['error'] is not None and not np.isnan(m['drr']['error']) else np.nan),
        ('C50 MAE',          'c50_abs_error',   'dB', lambda m: abs(m['c50']['error']) if m['c50']['error'] is not None and not np.isnan(m['c50']['error']) else np.nan),
        ('C80 MAE',          'c80_abs_error',   'dB', lambda m: abs(m['c80']['error']) if m['c80']['error'] is not None and not np.isnan(m['c80']['error']) else np.nan),
        ('LSD (broadband)',  'lsd',             'dB', lambda m: m['lsd']['broadband']),
        ('EDC Distance',     'edc_distance',    'dB²',lambda m: m['edc_distance']['broadband']),
        ('Cosine Similarity','cosine_similarity','',   lambda m: m['cosine_similarity']),
    ]

    # Header
    col_w = 22  # column width for values
    hdr = f"  {'Metric':20s} | {'Diffusion':>{col_w}s} | {'Baseline':>{col_w}s} | {'Base - Diff':>{col_w}s}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for name, agg_key, unit, extractor in metrics_table:
        # Diffusion column
        diff_stats = aggregate.get(agg_key, {})
        diff_val = diff_stats.get('mean', np.nan)

        # Baseline column
        base_stats = baseline_aggregate.get(agg_key, {})
        base_val = base_stats.get('mean', np.nan)

        # Compute per-sample Baseline - Diffusion difference
        paired_diffs = []
        for dm, bm in zip(diff_all_metrics, baseline_all_metrics):
            dv = extractor(dm)
            bv = extractor(bm)
            if dv is not None and bv is not None and not np.isnan(dv) and not np.isnan(bv):
                paired_diffs.append(bv - dv)

        if paired_diffs:
            delta_mean = np.mean(paired_diffs)
            delta_str = f"{delta_mean:+.4f} {unit}"
        else:
            delta_str = "N/A"

        # Format columns
        def _fmt(val, std):
            if val is None or np.isnan(val):
                return "N/A".rjust(col_w)
            s = f"{val:.4f}"
            if std is not None and not np.isnan(std):
                s += f" ± {std:.4f}"
            return s.rjust(col_w)

        diff_str = _fmt(diff_val, diff_stats.get('std', np.nan))
        base_str = _fmt(base_val, base_stats.get('std', np.nan))

        lines.append(f"  {name:20s} | {diff_str} | {base_str} | {delta_str:>{col_w}s}")

    return lines


def _format_t60_per_band_section(aggregate, data_info):
    """Return summary lines for T60 per-band % error, including a reliability warning."""
    band_agg = aggregate.get('t60_band_perc_error', {})
    if not band_agg:
        return []

    def fc_label(fc):
        return f"{int(fc)} Hz" if fc < 1000 else f"{int(fc / 1000)} kHz"

    lines = ["", "-" * 70, "T60 PER-BAND % ERROR (abs, mean ± std)", "-" * 70]
    sorted_bands = sorted(band_agg)
    for fc in sorted_bands:
        lines.append(format_metric_line(f"T60 {fc_label(fc):6s} %", band_agg[fc], '%'))

    # Reliability warning: flag bands where RIR is too short for the frequency
    MIN_CYCLES = 20
    rir_duration = data_info.get('sample_max_sec') if data_info else None
    if rir_duration:
        for fc in sorted_bands:
            n_cycles = rir_duration * fc
            if n_cycles < MIN_CYCLES:
                lines.append(f"  ! WARNING: {fc_label(fc)} — only {n_cycles:.0f} cycles in "
                             f"{rir_duration}s RIR (< {MIN_CYCLES}); T60 estimate may be unreliable.")
    return lines


def save_evaluation_summary(aggregate, n_samples, data_info, args, test_len, n_train_steps, save_path,
                            title="RIR DIFFUSION MODEL - EVALUATION SUMMARY",
                            baseline_aggregate=None, baseline_all_metrics=None, baseline_method=None,
                            all_metrics=None, t60_fit_range=None):
    """Save evaluation summary to a text file.

    Args:
        aggregate: Aggregated metrics dict
        n_samples: Number of samples evaluated
        data_info: Data info dict from model checkpoint
        args: Argument namespace with evaluation config
        test_len: Length of test set
        n_train_steps: Number of training timesteps
        save_path: Path to save the summary
        title: Title for the summary (default: real RIR evaluation)
        baseline_aggregate: Aggregated baseline metrics (optional)
        baseline_all_metrics: Per-sample baseline metrics list (optional, for paired comparison)
        baseline_method: Baseline method name (optional)
        all_metrics: Per-sample diffusion metrics list (needed for paired comparison table)
    """
    diff_all_metrics = all_metrics

    lines = [
        "=" * 70, title, "=" * 70,
        f"Timestamp: {get_israel_time('%Y-%m-%d %H:%M:%S')}",
        f"Total samples evaluated: {n_samples}",
        "", "-" * 70, "ACOUSTIC METRICS — Diffusion (Mean ± Std)", "-" * 70,
    ]

    # Diffusion metrics
    metrics_table = [
        ('T60 Error', 't60_error', 's'), ('T60 Abs Error', 't60_abs_error', 's'), ('T60 Perc Error', 't60_perc_error', '%'),
        ('EDT Error', 'edt_error', 's'), ('DRR Error', 'drr_error', 'dB'),
        ('DRR Abs Error', 'drr_abs_error', 'dB'),
        ('C50 MAE', 'c50_abs_error', 'dB'), ('C50 Bias', 'c50_error', 'dB'),
        ('C80 MAE', 'c80_abs_error', 'dB'), ('C80 Bias', 'c80_error', 'dB'),
        ('LSD (broadband)', 'lsd', 'dB'),
        ('EDC Distance', 'edc_distance', 'dB²'), ('Cosine Similarity', 'cosine_similarity', ''),
    ]
    for name, key, unit in metrics_table:
        if key in aggregate:
            lines.append(format_metric_line(name, aggregate[key], unit))

    lines.extend(_format_t60_per_band_section(aggregate, data_info))

    # Baseline section (if enabled)
    if baseline_aggregate is not None:
        baseline_label = 'pyroomacoustics' if baseline_method == 'pra' else 'RIR-Generator (Habets)'
        lines.extend([
            "", "-" * 70, f"ACOUSTIC METRICS — Baseline: {baseline_label} (Mean ± Std)", "-" * 70,
        ])
        for name, key, unit in metrics_table:
            if key in baseline_aggregate:
                lines.append(format_metric_line(name, baseline_aggregate[key], unit))

        lines.extend(_format_t60_per_band_section(baseline_aggregate, data_info))

        # Comparison table
        if baseline_all_metrics is not None and diff_all_metrics is not None:
            lines.extend([
                "", "-" * 70,
                "COMPARISON TABLE (Baseline - Diffusion, per-sample paired)",
                "  NOTE: For error metrics (lower=better), negative means diffusion did WORSE.",
                "        For Cosine Similarity (higher=better), negative means diffusion did WORSE.",
                "-" * 70,
            ])
            lines.extend(_format_comparison_table(aggregate, baseline_aggregate, diff_all_metrics, baseline_all_metrics))

    print("\n".join(lines))

    # Configuration and data info
    lines.extend([
        "", "=" * 70, "EVALUATION CONFIGURATION", "=" * 70,
        f"Model path: {args.model_path}",
        f"Dataset path: {getattr(args, 'dataset_path', None)}",
        f"Speech path (for LSD): {getattr(args, 'speech_path', None)}",
        f"Guidance scale: {args.guidance_scale}",
        f"Scheduler: {'DDIM' if getattr(args, 'use_ddim', False) else 'DDPM'}",
        f"Num inference steps: {args.num_inference_steps}",
        f"Num training steps: {n_train_steps}",
        f"Batch size: {getattr(args, 'batch_size', None)}",
        f"Octave bands: {getattr(args, 'octave_bands', None)}",
        f"Num workers: {getattr(args, 'workers', None)}",
        f"T60 fit range (dB): {t60_fit_range}",
    ])
    if baseline_method:
        lines.append(f"Baseline method: {baseline_method}")
        if data_info.get('dataset_name') == 'soundspaces':
            lines.append("  NOTE: Baseline RT60 values were estimated from the reference RIRs (SoundSpaces condition vector does not include RT60).")

    lines.extend([
        "", "-" * 70, "DATA INFO (from training)", "-" * 70,
        f"Test set size: {test_len}"
    ])
    for key, value in data_info.items():
        lines.append(f"  {key}: {value}")

    # Note about tail truncation during training
    if data_info.get('apply_zero_tail', False):
        lines.append(f"\n * NOTE: Training used tail truncation (db_cutoff={data_info.get('db_cutoff', -40)}dB), evaluation uses full reference RIRs")

    # Warning if RIR scaling was used during training
    if data_info.get('scale_rir', False):
        lines.extend([
            "", "!" * 70,
            "NOTE: RIR Scaling Applied During Training",
            "  Evaluation automatically unscales generated RIRs according to reference RIR energy decay",
            "!" * 70
        ])

    lines.append("=" * 70)

    # Save
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Summary saved to {save_path}")


def save_detailed_metrics_table(all_metrics, all_samples, save_path):
    """Save detailed metrics for each RIR to a CSV file.

    Args:
        all_metrics: List of metric dicts for each sample
        all_samples: List of sample dicts with condition info
        save_path: Path to save the CSV file
    """
    lines = []

    band_freqs = sorted(all_metrics[0]['t60'].get('per_band_perc', {}).keys()) if all_metrics else []

    def fc_col_name(fc):
        return f"T60_Perc_{int(fc)}Hz"

    # Header
    header = [
        "RIR_Index",
        "Room_Length", "Room_Width", "Room_Height",
        "Mic_X", "Mic_Y", "Mic_Z",
        "Speaker_X", "Speaker_Y", "Speaker_Z",
        "RT60_Target",
        "T60_Error", "T60_Abs_Error",
        "EDT_Error", "EDT_Abs_Error",
        "DRR_Error", "DRR_Abs_Error",
        "C50_Error", "C80_Error",
        "LSD", "EDC_Distance", "Cosine_Similarity",
    ] + [fc_col_name(fc) for fc in band_freqs]
    lines.append(",".join(header))

    # Data rows
    for idx, (metrics, sample) in enumerate(zip(all_metrics, all_samples)):
        cond = sample['condition']

        # Extract condition values (room_dim[3], mic_loc[3], speaker_loc[3], rt60[1])
        per_band_perc = metrics['t60'].get('per_band_perc', {})
        row = [
            str(idx),
            f"{cond[0]:.3f}", f"{cond[1]:.3f}", f"{cond[2]:.3f}",  # Room dimensions
            f"{cond[3]:.3f}", f"{cond[4]:.3f}", f"{cond[5]:.3f}",  # Mic location
            f"{cond[6]:.3f}", f"{cond[7]:.3f}", f"{cond[8]:.3f}",  # Speaker location
            f"{cond[9]:.3f}",  # RT60 target
            f"{metrics['t60']['broadband']:.4f}",
            f"{abs(metrics['t60']['broadband']):.4f}",
            f"{metrics['edt']['error']:.4f}",
            f"{abs(metrics['edt']['error']):.4f}",
            f"{metrics['drr']['error']:.4f}",
            f"{abs(metrics['drr']['error']):.4f}",
            f"{metrics['c50']['error']:.4f}",
            f"{metrics['c80']['error']:.4f}",
            f"{metrics['lsd']['broadband']:.4f}",
            f"{metrics['edc_distance']['broadband']:.4f}",
            f"{metrics['cosine_similarity']:.4f}",
        ] + [
            f"{per_band_perc[fc]:.4f}" if fc in per_band_perc else "nan"
            for fc in band_freqs
        ]
        lines.append(",".join(row))

    # Write to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Detailed metrics table saved to {save_path}")


def save_selected_samples(selected_samples, data_info, save_dir):
    """Save selected samples for each metric to separate files.

    Args:
        selected_samples: Dict from select_representative_samples
        data_info: Data info dict from model checkpoint
        save_dir: Directory to save the files

    Saved structure: {'metric_name': str, 'samples': {quality: [{'sample': {generated, reference, condition, metrics}, 'idx': int, 'value': float}, ...]}, 'data_info': dict}
                     Each quality category (best/worst/median/mean) contains a list of samples.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, samples_dict in selected_samples.items():
        # Save the samples dict along with data_info
        save_data = {
            'metric_name': metric_name,
            'samples': samples_dict,  # Keep the original structure
            'data_info': data_info
        }

        # Save to file
        filename = f'selected_samples_{metric_name}.pt'
        save_path = save_dir / filename
        torch.save(save_data, save_path)
        print(f"Selected samples for {metric_name} saved to: {save_path}")
