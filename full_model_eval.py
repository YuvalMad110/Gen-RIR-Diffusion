#!/usr/bin/env python3
"""
RIR Diffusion Model Evaluation Script

Comprehensive evaluation of generated RIRs against ground truth on the test set.
Computes acoustic metrics (T60, DRR, EDT, C50, LSD, EDC) and generates statistical reports.

Usage:
    python run_evaluation.py --model_path /path/to/model.pth.tar
    python run_evaluation.py --model_path /path/to/model.pth.tar --guidance_scale 4.0 --batch_size 32
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from RIRDiffusionModel import RIRDiffusionModel
from data.rir_dataset import load_rir_dataset
from data.dataset_collate_fn import scale_and_spectrogram_collate_fn
from utils.signal_proc import spectrogram_to_waveform, undo_rir_scaling, calculate_edc, estimate_decay_k_factor
from utils.inference_data_loading import load_model_and_data_info
from utils.acoustic_metrics import evaluate_rir_pair, aggregate_metrics, compute_edc, align_rir_lengths


def get_israel_timestamp():
    """Get current timestamp in Israel timezone."""
    israel_tz = pytz.timezone('Israel')
    return datetime.now(israel_tz).strftime("%Y-%m-%d_%H-%M-%S")


def get_israel_datetime_str():
    """Get formatted datetime string in Israel timezone."""
    israel_tz = pytz.timezone('Israel')
    return datetime.now(israel_tz).strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# Evaluation Core
# =============================================================================

def evaluate_test_set(model, test_dataloader, device, data_info, args):
    """Run evaluation over the entire test set."""
    model.eval()

    all_metrics = []
    sample_pairs = []

    sr = data_info['sr_target']
    hop_length = data_info['hop_length']
    n_fft = data_info['n_fft']
    sample_size = model.sample_size
    num_inference_steps = args.num_inference_steps
    octave_bands = args.octave_bands
    scale_rir = data_info.get('scale_rir', False)
    
    # Print evaluation configuration
    print(f"\nEvaluating on test set ({len(test_dataloader)} batches)...")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Scheduler: {'DDIM' if args.use_ddim else 'DDPM'}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Octave bands: {octave_bands}")

    n_samples_saved = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating batches"):
            # Unpack batch - default collate stacks the 5-tuple into 5 separate tensors
            # batch = (rirs_tensor, room_dims_tensor, mic_locs_tensor, speaker_locs_tensor, rt60s_tensor)
            rirs, room_dims, mic_locs, speaker_locs, rt60s = batch
            batch_size = rirs.shape[0]

            # Convert RIRs to numpy waveforms
            real_waveforms = []
            for i in range(batch_size):
                rir_np = rirs[i].cpu().numpy() if torch.is_tensor(rirs[i]) else rirs[i]
                real_waveforms.append(rir_np.squeeze())

            # Build condition tensor [B, 10]: room_dim(3) + mic_loc(3) + speaker_loc(3) + rt60(1)
            conditions = torch.cat([room_dims, mic_locs, speaker_locs, rt60s.unsqueeze(1)], dim=1).float()
            conditions = conditions.to(device)

            # Generate RIRs
            channels = 2  # Real + Imag for spectrogram
            shape = (batch_size, channels, *sample_size)

            generated_specs = model.generate(
                cond=conditions, shape=shape,
                num_inference_steps=num_inference_steps, guidance_scale=args.guidance_scale, verbose=False,
                use_ddim=args.use_ddim
            )

            if torch.is_tensor(generated_specs):
                generated_specs = generated_specs.cpu().numpy()

            # Convert generated spectrograms to waveforms
            gen_waveforms = [spectrogram_to_waveform(generated_specs[i], hop_length, n_fft) for i in range(batch_size)]

            # Reference waveforms are already in unscaled space (raw from dataset)
            # Generated waveforms need to be unscaled if scaling was used during training
            if scale_rir:
                # Convert to tensors
                real_tensor = torch.stack([torch.tensor(w, dtype=torch.float32) for w in real_waveforms])
                gen_tensor = torch.stack([torch.tensor(w, dtype=torch.float32) for w in gen_waveforms])

                # Compute k-factors from UNscaled reference RIRs (raw from dataset)
                edc = calculate_edc(real_tensor)
                k_factors, _ = estimate_decay_k_factor(edc, sr, data_info.get('db_cutoff', -40))

                # Undo scaling ONLY on generated RIRs (reference is already unscaled)
                gen_unscaled = undo_rir_scaling(gen_tensor, k_factors, sr)

                # Convert back to lists of numpy arrays
                gen_waveforms = [g.cpu().numpy() for g in gen_unscaled]
                # real_waveforms already unscaled, no change needed

            # Evaluate each pair and store all samples
            for i in range(batch_size):
                rir_gen, rir_ref = align_rir_lengths(gen_waveforms[i], real_waveforms[i], mode='truncate')
                metrics = evaluate_rir_pair(rir_gen, rir_ref, sr, octave_bands)
                all_metrics.append(metrics)

                # Store all samples for later selection
                sample_pairs.append({
                    'generated': rir_gen,
                    'reference': rir_ref,
                    'condition': conditions[i].cpu().numpy(),
                    'metrics': metrics
                })

    aggregate = aggregate_metrics(all_metrics)

    return aggregate, all_metrics, sample_pairs


def select_representative_samples(all_samples, metric_names):
    """Select best, worst, median, and mean-closest samples for each metric.

    Args:
        all_samples: List of dicts with 'generated', 'reference', 'condition', 'metrics'
        metric_names: List of metric names (e.g., ['t60', 'lsd'])

    Returns:
        dict: {metric_name: {'best': {'sample': ..., 'idx': ..., 'value': ...}, 'worst': {...}, ...}}
    """
    # Define how to extract each metric value from the metrics dict
    metric_extractors = {
        't60': lambda m: abs(m['t60']['broadband']) if not np.isnan(m['t60']['broadband']) else np.inf,
        'lsd': lambda m: m['lsd']['broadband'],
        'edt': lambda m: abs(m['edt']['broadband']) if not np.isnan(m['edt']['broadband']) else np.inf,
        'drr': lambda m: abs(m['drr']['broadband']) if not np.isnan(m['drr']['broadband']) else np.inf,
        'c50': lambda m: abs(m['c50']['broadband']) if not np.isnan(m['c50']['broadband']) else np.inf,
        'c80': lambda m: abs(m['c80']['broadband']) if not np.isnan(m['c80']['broadband']) else np.inf,
        'edc': lambda m: m['edc_distance']['broadband'],
        'cosine': lambda m: m['cosine_similarity'],
    }

    selected = {}

    for metric_name in metric_names:
        if metric_name not in metric_extractors:
            print(f"Warning: Unknown metric '{metric_name}', skipping")
            continue

        extractor = metric_extractors[metric_name]

        # Extract metric values from all samples (keep inf for worst, filter only nan for safety)
        values = []
        for sample in all_samples:
            val = extractor(sample['metrics'])
            # Keep inf values (they represent worst cases), but safety check for nan
            if not np.isnan(val):
                values.append(val)
            else:
                values.append(np.inf if metric_name != 'cosine' else -np.inf)

        values = np.array(values)

        # Find indices for best, worst, median, mean-closest
        # For most metrics, lower is better (except cosine where higher is better)
        if metric_name == 'cosine':
            best_idx = np.argmax(values)  # Highest similarity
            worst_idx = np.argmin(values)  # Lowest similarity
        else:
            best_idx = np.argmin(values)  # Lowest error
            worst_idx = np.argmax(values)  # Highest error (includes inf)

        # For median and mean, exclude inf values
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            sorted_indices = np.argsort(values)
            finite_sorted = sorted_indices[np.isfinite(values[sorted_indices])]
            median_idx = finite_sorted[len(finite_sorted) // 2]

            mean_val = np.mean(finite_values)
            mean_idx = np.argmin(np.abs(values - mean_val))
        else:
            # Fallback if all values are inf - use different indices
            median_idx = min(1, len(all_samples) - 1)
            mean_idx = min(2, len(all_samples) - 1)
            mean_val = np.inf

        selected[metric_name] = {
            'best': {'sample': all_samples[best_idx], 'idx': int(best_idx), 'value': float(values[best_idx])},
            'worst': {'sample': all_samples[worst_idx], 'idx': int(worst_idx), 'value': float(values[worst_idx])},
            'median': {'sample': all_samples[median_idx], 'idx': int(median_idx), 'value': float(values[median_idx])},
            'mean': {'sample': all_samples[mean_idx], 'idx': int(mean_idx), 'value': float(values[mean_idx])},
        }

        print(f"\n{metric_name} - Selected samples:")
        print(f"  Best:   idx={best_idx}, value={values[best_idx]:.4f}")
        print(f"  Worst:  idx={worst_idx}, value={values[worst_idx]:.4f}")
        print(f"  Median: idx={median_idx}, value={values[median_idx]:.4f}")
        print(f"  Mean:   idx={mean_idx}, value={values[mean_idx]:.4f} (target={mean_val:.4f})")

    return selected


# =============================================================================
# Visualization
# =============================================================================

def plot_metric_histogram(values, metric_name, unit, save_path, bins=30):
    """Plot histogram for a single metric."""
    values = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]
    if not values:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
    ax.axvline(np.median(values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.3f}')
    ax.set_xlabel(f'{metric_name} ({unit})', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{metric_name} Distribution (n={len(values)})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_histograms(all_metrics, save_dir):
    """Generate histograms for all metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = [
        ([r['t60']['broadband'] for r in all_metrics], 'T60 Error', 's', 'hist_t60_error.png'),
        ([abs(e) for e in [r['t60']['broadband'] for r in all_metrics] if e is not None and not np.isnan(e)], 'T60 Absolute Error', 's', 'hist_t60_abs_error.png'),
        ([r['drr']['error'] for r in all_metrics], 'DRR Error', 'dB', 'hist_drr_error.png'),
        ([r['edt']['error'] for r in all_metrics], 'EDT Error', 's', 'hist_edt_error.png'),
        ([r['c50']['error'] for r in all_metrics], 'C50 Error', 'dB', 'hist_c50_error.png'),
        ([r['c80']['error'] for r in all_metrics], 'C80 Error', 'dB', 'hist_c80_error.png'),
        ([r['lsd']['broadband'] for r in all_metrics], 'Log-Spectral Distance', 'dB', 'hist_lsd.png'),
        ([r['edc_distance']['broadband'] for r in all_metrics], 'EDC Distance (MSE)', 'dB²', 'hist_edc_distance.png'),
        ([r['cosine_similarity'] for r in all_metrics], 'Cosine Similarity', '', 'hist_cosine_similarity.png'),
    ]

    for values, name, unit, filename in metrics_to_plot:
        plot_metric_histogram(values, name, unit, save_dir / filename)

    print(f"Histograms saved to {save_dir}")


def plot_histograms_summary(all_metrics, n_samples, save_path):
    """Create a summary figure with all histograms in one image."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    metrics_data = [
        ([r['t60']['broadband'] for r in all_metrics], 'T60 Error (s)', 'T60'),
        ([abs(r['t60']['broadband']) for r in all_metrics if r['t60']['broadband'] is not None and not np.isnan(r['t60']['broadband'])], 'T60 Abs Error (s)', '|T60|'),
        ([r['drr']['error'] for r in all_metrics], 'DRR Error (dB)', 'DRR'),
        ([r['edt']['error'] for r in all_metrics], 'EDT Error (s)', 'EDT'),
        ([r['c50']['error'] for r in all_metrics], 'C50 Error (dB)', 'C50'),
        ([r['c80']['error'] for r in all_metrics], 'C80 Error (dB)', 'C80'),
        ([r['lsd']['broadband'] for r in all_metrics], 'LSD (dB)', 'LSD'),
        ([r['edc_distance']['broadband'] for r in all_metrics], 'EDC Dist (dB²)', 'EDC'),
        ([r['cosine_similarity'] for r in all_metrics], 'Cosine Similarity', 'Cos Sim'),
    ]

    for ax, (values, ylabel, title) in zip(axes.flatten(), metrics_data):
        values = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]
        if values:
            ax.hist(values, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=1.5)
            ax.set_title(f'{title}\nμ={np.mean(values):.3f}, σ={np.std(values):.3f}', fontsize=10)
            ax.set_xlabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Evaluation Summary (n={n_samples})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary figure saved to {save_path}")


def plot_selected_rir_samples(selected_samples, metric_name, sr, model_name, n_timesteps_inference, n_timesteps_train, save_dir):
    """Plot all selected RIR pairs on a single image with separate waveforms, EDC, and metrics.

    Args:
        selected_samples: Dict of all selected samples (output from select_representative_samples)
        metric_name: Name of the metric to plot (e.g., 't60', 'lsd')
        sr: Sample rate
        model_name: Name of the model (folder name)
        n_timesteps_inference: Number of inference timesteps used
        n_timesteps_train: Number of training timesteps
        save_dir: Directory to save the plot (filename will be generated automatically)
    """
    # Extract the samples dict for this metric
    samples_dict = selected_samples[metric_name]

    # Order: best, worst, median, mean
    quality_order = ['best', 'worst', 'median', 'mean']
    n_pairs = len(quality_order)

    # Each row: waveform_real, waveform_gen, EDC, text (metrics + conditions combined)
    fig, axes = plt.subplots(n_pairs, 4, figsize=(20, 3.5 * n_pairs))

    # Main title with metric name
    metric_display = metric_name.upper() if metric_name in ['lsd', 'edc', 'drr'] else metric_name.capitalize()
    main_title = f"Model: {model_name} | Metric: {metric_display} | Inference Steps: {n_timesteps_inference} | Training Steps: {n_timesteps_train}"
    fig.suptitle(main_title, fontsize=14, fontweight='bold')

    for i, quality_label in enumerate(quality_order):
        pair = samples_dict[quality_label]['sample']

        rir_gen = pair['generated']
        rir_ref = pair['reference']
        condition = pair['condition']
        metrics = pair['metrics']

        # Extract condition info
        room_dims = condition[:3]
        rt60_real = condition[-1]
        mic_loc = condition[3:6]
        speaker_loc = condition[6:9]
        speaker_mic_dist = np.linalg.norm(speaker_loc - mic_loc)

        # Row title
        row_title = f"{quality_label.capitalize()}: Spk-Mic {speaker_mic_dist:.2f}m"

        # Calculate max time for consistent x-axis
        t_ref = np.arange(len(rir_ref)) / sr * 1000
        t_gen = np.arange(len(rir_gen)) / sr * 1000
        max_time = max(t_ref[-1], t_gen[-1])

        # --- Column 0: Real Waveform ---
        axes[i, 0].plot(t_ref, rir_ref, color='green', linewidth=0.8)
        axes[i, 0].set_ylabel('Amplitude', fontsize=9)
        axes[i, 0].set_title(f'{row_title} - Real', fontsize=10, fontweight='bold', loc='left')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim([0, max_time])
        if i == n_pairs - 1:
            axes[i, 0].set_xlabel('Time (ms)', fontsize=9)

        # --- Column 1: Generated Waveform ---
        axes[i, 1].plot(t_gen, rir_gen, color='blue', linewidth=0.8)
        axes[i, 1].set_ylabel('Amplitude', fontsize=9)
        axes[i, 1].set_title('Generated', fontsize=10, fontweight='bold', loc='left')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim([0, max_time])
        if i == n_pairs - 1:
            axes[i, 1].set_xlabel('Time (ms)', fontsize=9)

        # --- Column 2: EDC ---
        edc_gen = compute_edc(rir_gen)
        edc_ref = compute_edc(rir_ref)
        t_edc = np.arange(len(edc_gen)) / sr * 1000
        axes[i, 2].plot(t_edc, edc_ref, label='Real', color='green', linewidth=1.2)
        axes[i, 2].plot(t_edc, edc_gen, label='Generated', color='blue', linewidth=1.2, linestyle='--')
        axes[i, 2].axhline(y=-60, color='red', linestyle=':', alpha=0.5, linewidth=1)
        axes[i, 2].set_ylabel('Energy (dB)', fontsize=9)
        axes[i, 2].set_title('Energy Decay Curve', fontsize=10)
        axes[i, 2].legend(loc='upper right', fontsize=8)
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_ylim([-80, 5])
        if i == n_pairs - 1:
            axes[i, 2].set_xlabel('Time (ms)', fontsize=9)

        # --- Column 3: Metrics + Conditions Text ---
        axes[i, 3].axis('off')

        # Metrics text
        m = metrics
        metrics_text = (
            f"METRICS:\n"
            f"  T60 Error: {m['t60']['broadband']:.3f} s\n"
            f"  EDT Error: {m['edt']['error']:.3f} s\n"
            f"  DRR Error: {m['drr']['error']:.2f} dB\n"
            f"  C50 Error: {m['c50']['error']:.2f} dB\n"
            f"  C80 Error: {m['c80']['error']:.2f} dB\n"
            f"  LSD: {m['lsd']['broadband']:.2f} dB\n"
            f"  EDC Dist: {m['edc_distance']['broadband']:.3f}\n"
            f"  Cosine Sim: {m['cosine_similarity']:.4f}\n"
        )

        # Conditions text
        conditions_text = (
            f"\nCONDITIONS:\n"
            f"  Room: {room_dims[0]:.1f} x {room_dims[1]:.1f} x {room_dims[2]:.1f} m\n"
            f"  RT60: {rt60_real:.2f} s\n"
            f"  Mic: ({mic_loc[0]:.2f}, {mic_loc[1]:.2f}, {mic_loc[2]:.2f})\n"
            f"  Speaker: ({speaker_loc[0]:.2f}, {speaker_loc[1]:.2f}, {speaker_loc[2]:.2f})\n"
            f"  Distance: {speaker_mic_dist:.2f} m"
        )

        full_text = metrics_text + conditions_text
        axes[i, 3].text(0.05, 0.5, full_text, transform=axes[i, 3].transAxes,
                       fontsize=9, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save with metric name in filename
    save_path = Path(save_dir) / f'{metric_name}_comparison.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"RIR comparison plot for {metric_name} saved to: {save_path}")


# =============================================================================
# Reporting
# =============================================================================

def format_metric_line(name, stats, unit):
    """Format a single metric line for the summary."""
    if stats['mean'] is None or np.isnan(stats['mean']):
        return f"  {name:20s}: N/A"
    return f"  {name:20s}: {stats['mean']:8.4f} ± {stats['std']:.4f} {unit}  (median: {stats['median']:.4f}, n_valid={stats['n_valid']})"


def save_evaluation_summary(aggregate, n_samples, data_info, args, test_len, n_train_steps, save_path):
    """Save evaluation summary to a text file."""
    lines = [
        "=" * 70, "RIR DIFFUSION MODEL - EVALUATION SUMMARY", "=" * 70,
        f"Timestamp: {get_israel_datetime_str()}",
        f"Total samples evaluated: {n_samples}",
        "", "-" * 70, "ACOUSTIC METRICS (Mean ± Std)", "-" * 70, ]

    # Metrics
    metrics_table = [
        ('T60 Error', 't60_error', 's'), ('T60 Abs Error', 't60_abs_error', 's'),
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
        f"Dataset path: {args.dataset_path}",
        f"Guidance scale: {args.guidance_scale}",
        f"Scheduler: {'DDIM' if args.use_ddim else 'DDPM'}",
        f"Num inference steps: {args.num_inference_steps}",
        f"Num training steps: {n_train_steps}",
        f"Batch size: {args.batch_size}",
        f"Octave bands: {args.octave_bands}",
        f"Num workers: {args.workers}",
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

    Saved structure: {'metric_name': str, 'samples': {quality: {'sample': {generated, reference, condition, metrics}, 'idx': int, 'value': float}}, 'data_info': dict}
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


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RIR Diffusion Model on Test Set")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint",
                        default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/Aug12_12-09-09_dgx03/model_best.pth.tar')
    parser.add_argument("--dataset_path", type=str, default='./datasets/GTU_rir/GTU_RIR.pickle.dat', help="Path to RIR dataset")
    parser.add_argument("--nSamples", type=int, default=None, help="Number of samples (None=use all from data_info)")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="CFG scale (1.0=no guidance)")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Denoising steps (None=use training timesteps)")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--octave_bands", type=float, nargs='+', default=[125, 250, 500, 1000, 2000, 4000], help="Octave bands")
    parser.add_argument("--save_path", type=str, default=None, help="Output directory")
    parser.add_argument("--n_examples", type=int, default=5, help="Number of example pairs to visualize")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--debug_mode", type=bool, default=False, help="Debug mode: fast run")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # ---------- Load model and data_info ----------
    print("\nLoading model...")
    model, data_info = load_model_and_data_info(args.model_path, device, RIRDiffusionModel)
    args.num_inference_steps = args.num_inference_steps or model.n_timesteps

    # Debug mode overrides
    if args.debug_mode:
        print("\n[DEBUG MODE] Overriding settings for fast run!!!")
        args.nSamples = 4
        args.num_inference_steps = 2
        args.batch_size = 4

    # Use seed from training for reproducible test split
    torch.manual_seed(data_info['random_seed'])
    np.random.seed(data_info['random_seed'])

    # Calculate total nSamples from desired test set size
    # User specifies test set size, we back-calculate total: total = test_size / test_ratio
    if args.nSamples is not None:
        n_samples = int(args.nSamples / data_info['test_ratio'])
        print(f"Test set size requested: {args.nSamples} -> Total dataset size: {n_samples}")
    else:
        n_samples = data_info['nSamples']
    
    # Check CFG
    if args.guidance_scale != 1.0 and not getattr(model, 'guidance_enabled', False):
        print("Warning: Model was not trained with CFG. Setting guidance_scale to 1.0")
        args.guidance_scale = 1.0
    
    # Output directory
    if args.save_path is None:
        folder_name = f"evaluation_{get_israel_timestamp()}"
        if getattr(model, 'guidance_enabled', False):
            folder_name += f"_guidance{args.guidance_scale}"
        args.save_path = os.path.join(os.path.dirname(args.model_path), folder_name)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # ---------- Load test dataset ----------
    print("\nLoading test dataset...")
    _, _, test_dataset = load_rir_dataset(
        name='gtu', path=args.dataset_path, split=True, mode='raw',
        hop_length=data_info['hop_length'], n_fft=data_info['n_fft'],
        use_spectrogram=data_info['use_spectrogram'], sample_max_sec=data_info['sample_max_sec'],
        nSamples=n_samples, sr_target=data_info['sr_target'],
        train_ratio=data_info['train_ratio'], eval_ratio=data_info['eval_ratio'],
        test_ratio=data_info['test_ratio'], random_seed=data_info['random_seed'],
        split_by_room=data_info['split_by_room']
    )
    print(f"Test set size: {len(test_dataset)}")
    
    # Create dataloader WITHOUT collate function to get raw unscaled waveforms
    # We'll handle any needed transformations manually in evaluate_test_set
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=None, drop_last=False, pin_memory=torch.cuda.is_available()
    )
    
    # ---------- Run evaluation ----------
    aggregate, all_metrics, all_samples = evaluate_test_set(model, test_dataloader, device, data_info, args)

    # Select representative samples based on hardcoded metrics
    print("\nSelecting representative samples...")
    metric_names = ['t60', 'lsd']  # Hardcoded: T60 Abs Error and LSD
    selected_samples = select_representative_samples(all_samples, metric_names)

    # ---------- Reporting and Visualization ----------
    # Print and save results
    save_evaluation_summary(aggregate, len(all_metrics), data_info, args, len(test_dataset), model.n_timesteps, save_path / 'evaluation_summary.txt')
    save_detailed_metrics_table(all_metrics, all_samples, save_path / 'detailed_metrics.csv')
    save_selected_samples(selected_samples, data_info, save_path)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_histograms_summary(all_metrics, len(all_metrics), save_path / 'histograms_summary.png')
    plot_all_histograms(all_metrics, save_path / 'histograms')

    # Plot selected representative samples (4 per metric)
    # Extract model name from model path (parent folder name)
    model_name = Path(args.model_path).parent.name
    for metric_name in selected_samples.keys():
        plot_selected_rir_samples(selected_samples, metric_name, data_info['sr_target'],
                                  model_name, args.num_inference_steps, model.n_timesteps, save_path)

    print(f"\n✓ Evaluation complete! Results saved to: {save_path}")


if __name__ == "__main__":
    main()