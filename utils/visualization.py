"""
Visualization utilities for RIR inference and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict

from utils.acoustic_metrics import compute_edc
from utils.signal_edc import create_edc_plots_mode2


def format_condition_string(condition: np.ndarray, mode: str = "room") -> str:
    """Format condition parameters for display."""
    if mode == "room":
        room_dims = condition[:3]
        rt60 = condition[-1]
        return f"Room: {room_dims[0]:.1f}x{room_dims[1]:.1f}x{room_dims[2]:.1f}m, RT60: {rt60:.2f}s"
    elif mode == "locations":
        mic_loc = condition[3:6]
        speaker_loc = condition[6:9]
        return f"Mic: ({mic_loc[0]:.1f},{mic_loc[1]:.1f},{mic_loc[2]:.1f}), Src: ({speaker_loc[0]:.1f},{speaker_loc[1]:.1f},{speaker_loc[2]:.1f})"
    return ""


def plot_comparison(real_rirs_wave: List[np.ndarray], generated_rirs_wave: List[np.ndarray],
                    real_rirs_spec: List[np.ndarray], generated_rirs_spec: List[np.ndarray],
                    conditions: np.ndarray, rir_indices: List[int], sr: int, save_path: str,
                    title: str = "", metrics: Optional[Dict] = None):
    """Create comparison plot: real vs generated (waveforms and spectrograms)."""
    n_rirs = len(real_rirs_wave)
    fig, axes = plt.subplots(n_rirs, 4, figsize=(20, 4 * n_rirs))
    
    if title:
        fig.suptitle(title, fontsize=12)
    if n_rirs == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_rirs):
        idx = rir_indices[i]
        real_wave, gen_wave = real_rirs_wave[i], generated_rirs_wave[i]
        real_spec, gen_spec = real_rirs_spec[i], generated_rirs_spec[i]
        
        condition_str = format_condition_string(conditions[i], "room")
        location_str = format_condition_string(conditions[i], "locations")
        
        # Real waveform
        time_real = np.arange(len(real_wave)) / sr
        axes[i, 0].plot(time_real, real_wave, linewidth=0.8, color='green')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'Real RIR #{idx} - {condition_str}')
        
        # Generated waveform
        time_gen = np.arange(len(gen_wave)) / sr
        axes[i, 1].plot(time_gen, gen_wave, linewidth=0.8, color='blue')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title(f'Generated RIR #{idx}')
        
        # Real spectrogram
        real_mag = np.sqrt(real_spec[0]**2 + real_spec[1]**2)
        axes[i, 2].imshow(20 * np.log10(real_mag + 1e-8), aspect='auto', origin='lower', cmap='viridis')
        axes[i, 2].set_title(f'Real RIR #{idx} Spectrogram')
        axes[i, 2].set_ylabel('Frequency Bin')
        
        # Generated spectrogram
        gen_mag = np.sqrt(gen_spec[0]**2 + gen_spec[1]**2)
        axes[i, 3].imshow(20 * np.log10(gen_mag + 1e-8), aspect='auto', origin='lower', cmap='viridis')
        axes[i, 3].set_title(f'Generated RIR #{idx} Spectrogram - {location_str}')
        axes[i, 3].set_ylabel('Frequency Bin')
        
        if i == n_rirs - 1:
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 2].set_xlabel('Time Frame')
            axes[i, 3].set_xlabel('Time Frame')
    
    plt.tight_layout()
    plot_path = save_path if save_path.endswith(".png") else str(Path(save_path) / "rirs_comparison.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved to: {plot_path}")


def plot_edc_comparison(real_rirs: List[np.ndarray], generated_rirs: List[np.ndarray],
                        conditions: np.ndarray, rir_indices: List[int], sr: int, save_path: str,
                        metrics: Optional[Dict] = None, octave_bands: List[float] = None,
                        title: str = ""):
    """Create Energy Decay Curve comparison plot."""
    n_rirs = len(real_rirs)
    fig, axes = plt.subplots(n_rirs, 1, figsize=(12, 4 * n_rirs))
    
    if title:
        fig.suptitle(title, fontsize=12)
    if n_rirs == 1:
        axes = [axes]
    
    for i in range(n_rirs):
        idx = rir_indices[i]
        real_edc = _calculate_edc_numpy(real_rirs[i])
        gen_edc = _calculate_edc_numpy(generated_rirs[i])
        
        time_real = np.arange(len(real_edc)) / sr
        time_gen = np.arange(len(gen_edc)) / sr
        
        axes[i].plot(time_real, 10 * np.log10(real_edc + 1e-10), 
                     'g-', linewidth=1.5, label='Real', alpha=0.8)
        axes[i].plot(time_gen, 10 * np.log10(gen_edc + 1e-10), 
                     'b--', linewidth=1.5, label='Generated', alpha=0.8)
        
        rt60 = conditions[i, -1]
        axes[i].axhline(y=-60, color='r', linestyle=':', alpha=0.5, label='RT60 threshold')
        axes[i].axvline(x=rt60, color='orange', linestyle=':', alpha=0.5, label=f'RT60={rt60:.2f}s')
        
        condition_str = format_condition_string(conditions[i], "room")
        axes[i].set_title(f'EDC Comparison #{idx} - {condition_str}')
        axes[i].set_ylabel('Energy (dB)')
        axes[i].set_ylim(-80, 5)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')
        
        if i == n_rirs - 1:
            axes[i].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plot_path = save_path if save_path.endswith(".png") else str(Path(save_path) / "edc_comparison.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"EDC comparison plot saved to: {plot_path}")


def _calculate_edc_numpy(rir: np.ndarray) -> np.ndarray:
    """Calculate Energy Decay Curve from RIR."""
    energy = rir ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    return edc / (edc[0] + 1e-10)


# =============================================================================
# Evaluation Visualization Functions
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

    # Get number of samples per category (should be same for all categories)
    n_samples_per_category = len(samples_dict['best'])
    n_total_rows = len(quality_order) * n_samples_per_category

    # Each row: waveform_real, waveform_gen, normalized_overlay, EDC, text (metrics + conditions combined)
    fig, axes = plt.subplots(n_total_rows, 5, figsize=(24, 3.0 * n_total_rows))

    # Main title with metric name
    metric_display = metric_name.upper() if metric_name in ['lsd', 'edc', 'drr'] else metric_name.capitalize()
    main_title = f"Model: {model_name} | Metric: {metric_display} | Inference Steps: {n_timesteps_inference} | Training Steps: {n_timesteps_train}"
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=1.0)

    row_idx = 0
    for quality_label in quality_order:
        samples_list = samples_dict[quality_label]

        for sample_idx, sample_entry in enumerate(samples_list):
            pair = sample_entry['sample']
            metric_value = sample_entry['value']

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

            # Row title - include sample number within category
            row_title = f"{quality_label.capitalize()} #{sample_idx + 1}: {metric_display}={metric_value:.4f}"

            # Calculate max time for consistent x-axis
            t_ref = np.arange(len(rir_ref)) / sr * 1000
            t_gen = np.arange(len(rir_gen)) / sr * 1000
            max_time = max(t_ref[-1], t_gen[-1])

            # --- Column 0: Real Waveform ---
            axes[row_idx, 0].plot(t_ref, rir_ref, color='green', linewidth=0.8)
            axes[row_idx, 0].set_ylabel('Amplitude', fontsize=9)
            axes[row_idx, 0].set_title(f'{row_title} - Real', fontsize=10, fontweight='bold', loc='left')
            axes[row_idx, 0].grid(True, alpha=0.3)
            axes[row_idx, 0].set_xlim([0, max_time])
            if row_idx == n_total_rows - 1:
                axes[row_idx, 0].set_xlabel('Time (ms)', fontsize=9)

            # --- Column 1: Generated Waveform ---
            axes[row_idx, 1].plot(t_gen, rir_gen, color='blue', linewidth=0.8)
            axes[row_idx, 1].set_ylabel('Amplitude', fontsize=9)
            axes[row_idx, 1].set_title('Generated', fontsize=10, fontweight='bold', loc='left')
            axes[row_idx, 1].grid(True, alpha=0.3)
            axes[row_idx, 1].set_xlim([0, max_time])
            if row_idx == n_total_rows - 1:
                axes[row_idx, 1].set_xlabel('Time (ms)', fontsize=9)

            # --- Column 2: L2 Normalized Overlay ---
            # L2 normalize both RIRs
            rir_ref_norm = rir_ref / (np.linalg.norm(rir_ref) + 1e-10)
            rir_gen_norm = rir_gen / (np.linalg.norm(rir_gen) + 1e-10)
            axes[row_idx, 2].plot(t_ref, rir_ref_norm, color='green', linewidth=0.8, label='Real', alpha=0.7)
            axes[row_idx, 2].plot(t_gen, rir_gen_norm, color='blue', linewidth=0.8, label='Generated', alpha=0.7)
            axes[row_idx, 2].set_ylabel('Normalized Amp.', fontsize=9)
            axes[row_idx, 2].set_title('L2 Normalized Overlay', fontsize=10, fontweight='bold', loc='left')
            axes[row_idx, 2].legend(loc='upper right', fontsize=8)
            axes[row_idx, 2].grid(True, alpha=0.3)
            axes[row_idx, 2].set_xlim([0, max_time])
            if row_idx == n_total_rows - 1:
                axes[row_idx, 2].set_xlabel('Time (ms)', fontsize=9)

            # --- Column 3: EDC ---
            edc_gen = compute_edc(rir_gen)
            edc_ref = compute_edc(rir_ref)
            t_edc = np.arange(len(edc_gen)) / sr * 1000
            axes[row_idx, 3].plot(t_edc, edc_ref, label='Real', color='green', linewidth=1.2)
            axes[row_idx, 3].plot(t_edc, edc_gen, label='Generated', color='blue', linewidth=1.2, linestyle='--')
            axes[row_idx, 3].axhline(y=-40, color='yellow', linestyle=':', alpha=0.8, linewidth=1.2)
            axes[row_idx, 3].set_ylabel('Energy (dB)', fontsize=9)
            axes[row_idx, 3].set_title('Energy Decay Curve', fontsize=10)
            axes[row_idx, 3].legend(loc='upper right', fontsize=8)
            axes[row_idx, 3].grid(True, alpha=0.3)
            axes[row_idx, 3].set_ylim([-80, 5])
            if row_idx == n_total_rows - 1:
                axes[row_idx, 3].set_xlabel('Time (ms)', fontsize=9)

            # --- Column 4: Metrics + Conditions Text ---
            axes[row_idx, 4].axis('off')

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
            axes[row_idx, 4].text(0.05, 0.5, full_text, transform=axes[row_idx, 4].transAxes,
                           fontsize=9, verticalalignment='center', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            row_idx += 1

    plt.tight_layout()

    # Save with metric name in filename
    save_path = Path(save_dir) / f'{metric_name}_comparison.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"RIR comparison plot for {metric_name} saved to: {save_path}")


def plot_edc_per_band_for_selected(selected_samples, metric_name, sr, octave_bands, save_dir):
    """Plot EDC per octave band comparison for selected samples.

    Uses create_edc_plots_mode2 from utils/signal_edc.py to generate EDC plots.
    Creates a separate plot for each quality category (best, worst, median, mean).

    Args:
        selected_samples: Dict from select_representative_samples
        metric_name: Name of the metric (e.g., 't60', 'lsd')
        sr: Sample rate
        octave_bands: List of octave center frequencies
        save_dir: Directory to save the plots
    """
    samples_dict = selected_samples[metric_name]
    quality_order = ['best', 'worst', 'median', 'mean']

    # Create output directory
    edc_dir = Path(save_dir) / 'edc_per_band'
    edc_dir.mkdir(parents=True, exist_ok=True)

    metric_display = metric_name.upper() if metric_name in ['lsd', 'edc', 'drr'] else metric_name.capitalize()

    for quality_label in quality_order:
        samples_list = samples_dict[quality_label]

        # Extract samples for this quality category
        real_rirs_wave = []
        gen_rirs_wave = []
        conditions = []
        rir_indices = []

        for sample_idx, sample_entry in enumerate(samples_list):
            pair = sample_entry['sample']
            real_rirs_wave.append(pair['reference'])
            gen_rirs_wave.append(pair['generated'])
            conditions.append(pair['condition'])
            rir_indices.append(f"{quality_label}_{sample_idx + 1}")

        conditions = np.array(conditions)

        # Build title and save path
        title = f"EDC per Band - {metric_display} - {quality_label.capitalize()}"
        save_path = edc_dir / f'{metric_name}_{quality_label}_edc_per_band.png'

        # Call existing function
        create_edc_plots_mode2(
            real_rirs_wave, gen_rirs_wave, conditions, rir_indices,
            sr, str(save_path), metrics=None, octaves=octave_bands, title=title
        )
