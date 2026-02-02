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


def save_evaluation_summary(aggregate, n_samples, data_info, args, test_len, n_train_steps, save_path,
                            title="RIR DIFFUSION MODEL - EVALUATION SUMMARY"):
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
    """
    lines = [
        "=" * 70, title, "=" * 70,
        f"Timestamp: {get_israel_time('%Y-%m-%d %H:%M:%S')}",
        f"Total samples evaluated: {n_samples}",
        "", "-" * 70, "ACOUSTIC METRICS (Mean ± Std)", "-" * 70,
    ]

    # Metrics
    metrics_table = [
        ('T60 Error', 't60_error', 's'), ('T60 Abs Error', 't60_abs_error', 's'), ('T60 Perc Error', 't60_perc_error', '%'),
        ('EDT Error', 'edt_error', 's'), ('DRR Error', 'drr_error', 'dB'),
        ('DRR Abs Error', 'drr_abs_error', 'dB'), ('C50 Error', 'c50_error', 'dB'),
        ('C80 Error', 'c80_error', 'dB'), ('LSD (broadband)', 'lsd', 'dB'),
        ('EDC Distance', 'edc_distance', 'dB²'), ('Cosine Similarity', 'cosine_similarity', ''),
    ]
    for name, key, unit in metrics_table:
        if key in aggregate:
            lines.append(format_metric_line(name, aggregate[key], unit))
    print("\n".join(lines))

    # Configuration and data info
    lines.extend([
        "", "=" * 70, "EVALUATION CONFIGURATION", "=" * 70,
        f"Model path: {args.model_path}",
    ])

    # Add dataset path if available
    if hasattr(args, 'dataset_path'):
        lines.append(f"Dataset path: {args.dataset_path}")

    lines.extend([
        f"Guidance scale: {args.guidance_scale}",
        f"Scheduler: {'DDIM' if getattr(args, 'use_ddim', False) else 'DDPM'}",
        f"Num inference steps: {args.num_inference_steps}",
        f"Num training steps: {n_train_steps}",
        f"Batch size: {getattr(args, 'batch_size', 'N/A')}",
        f"Octave bands: {getattr(args, 'octave_bands', 'N/A')}",
    ])

    if hasattr(args, 'workers'):
        lines.append(f"Num workers: {args.workers}")

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
        "LSD", "EDC_Distance", "Cosine_Similarity"
    ]
    lines.append(",".join(header))

    # Data rows
    for idx, (metrics, sample) in enumerate(zip(all_metrics, all_samples)):
        cond = sample['condition']

        # Extract condition values (room_dim[3], mic_loc[3], speaker_loc[3], rt60[1])
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
            f"{metrics['cosine_similarity']:.4f}"
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
