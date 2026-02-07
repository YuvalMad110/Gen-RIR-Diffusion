"""
Dataset analysis script for GTU RIR dataset.
Computes metrics, generates histograms, and saves summary statistics.

Usage:
    python data/analyze_dataset.py --debug true   # Run on 10 samples only
    python data/analyze_dataset.py                # Run on full dataset
"""
import os
import sys
import argparse
import platform
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.gtu_rir import create_gtu_datasets
from utils.misc import str2bool, get_israel_time
from utils.acoustic_metrics import has_secondary_peak_artifact

# Metric units mapping (add new metrics here)
METRIC_UNITS = {
    'rt60': 's',
    'room_volume': 'm³',
    'source_target_distance': 'm',
}


def get_datasets_folder():
    """Returns the path to the dataset based on the platform."""
    if platform.system() == "Windows":
        return os.path.normpath('C:/Yuval/MSc/AV_RIR/code_exp/rir_encoder/data/GTU_RIR_1024samples.pickle.dat')
    else:
        return os.path.normpath('./datasets/GTU_rir/GTU_RIR.pickle.dat')


def compute_metrics(dataset, sr, output_dir, n_abnormal_plots=10):
    """
    Compute metrics for all samples in the dataset.

    Args:
        dataset: The dataset to analyze
        sr: Sample rate of the RIRs
        output_dir: Directory to save artifact plots
        n_abnormal_plots: Maximum number of abnormal RIR plots to save

    Returns:
        dict: Dictionary with metric names as keys and lists of values.
        dict: Artifact detection results with indices and count.
    """
    metrics = {
        'rt60': [],
        'room_volume': [],
        'source_target_distance': [],
    }

    # Artifact detection tracking
    artifact_indices = []
    n_plots_saved = 0

    # Create subfolder for artifact plots
    artifact_plot_dir = os.path.join(output_dir, 'secondary_peak_artifacts')
    os.makedirs(artifact_plot_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Computing metrics"):
        rir, room_dim, mic_loc, speaker_loc, rt60 = dataset[idx]

        # Convert RIR tensor to numpy if needed
        rir_np = rir.squeeze().numpy() if hasattr(rir, 'numpy') else np.asarray(rir).flatten()

        # RT60 (already in the dataset)
        metrics['rt60'].append(rt60)

        # Room volume [m^3] = depth * width * height
        volume = room_dim[0] * room_dim[1] * room_dim[2]
        metrics['room_volume'].append(volume)

        # Source-target (speaker-mic) distance [m]
        distance = np.linalg.norm(speaker_loc - mic_loc)
        metrics['source_target_distance'].append(distance)

        # Secondary peak artifact detection (fast check without plotting)
        has_artifact = has_secondary_peak_artifact(rir_np, sr)
        if has_artifact:
            artifact_indices.append(idx)
            # Save plot only if we haven't reached the limit
            if n_plots_saved < n_abnormal_plots:
                save_path = os.path.join(artifact_plot_dir, f'artifact_idx_{idx:06d}.png')
                has_secondary_peak_artifact(rir_np, sr, save_path=save_path)
                n_plots_saved += 1

    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])

    artifact_results = {
        'indices': artifact_indices,
        'count': len(artifact_indices),
        'total': len(dataset),
        'percentage': len(artifact_indices) / len(dataset) * 100,
        'n_plots_saved': n_plots_saved,
    }

    return metrics, artifact_results


def compute_statistics(values):
    """Compute summary statistics for a metric."""
    return {
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'count': len(values),
    }


def save_histogram(values, metric_name, output_dir, units=""):
    """Save histogram for a metric."""
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, edgecolor='black', alpha=0.7)

    title = f'{metric_name.replace("_", " ").title()} Distribution'
    xlabel = f'{metric_name.replace("_", " ").title()}'
    if units:
        xlabel += f' [{units}]'

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats = compute_statistics(values)
    stats_text = f"Mean: {stats['mean']:.3f}\nMedian: {stats['median']:.3f}\nStd: {stats['std']:.3f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    filename = os.path.join(output_dir, f'{metric_name}_histogram.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved histogram: {filename}")


def save_summary(metrics, artifact_results, output_dir, debug_mode):
    """Save summary statistics to a text file."""
    timestamp = get_israel_time("%Y-%m-%d %H:%M:%S")

    filename = os.path.join(output_dir, 'summary.txt')

    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GTU RIR Dataset Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Debug mode: {debug_mode}\n")
        f.write(f"Total samples analyzed: {len(metrics['rt60'])}\n\n")

        # Artifact detection results
        f.write("-" * 60 + "\n")
        f.write("Secondary Peak Artifact Detection\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Artifacts found: {artifact_results['count']} / {artifact_results['total']}\n")
        f.write(f"  Percentage:      {artifact_results['percentage']:.2f}%\n")
        f.write(f"  Plots saved:     {artifact_results['n_plots_saved']}\n")
        if artifact_results['indices']:
            indices_str = ', '.join(map(str, artifact_results['indices'][:20]))
            if len(artifact_results['indices']) > 20:
                indices_str += f", ... ({len(artifact_results['indices']) - 20} more)"
            f.write(f"  Artifact indices: {indices_str}\n")
        f.write("\n")

        for metric_name, values in metrics.items():
            stats = compute_statistics(values)
            units = METRIC_UNITS.get(metric_name, '')
            unit_str = f" [{units}]" if units else ""

            f.write("-" * 60 + "\n")
            f.write(f"Metric: {metric_name.replace('_', ' ').title()}{unit_str}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Min:    {stats['min']:.4f}\n")
            f.write(f"  Max:    {stats['max']:.4f}\n")
            f.write(f"  Mean:   {stats['mean']:.4f}\n")
            f.write(f"  Median: {stats['median']:.4f}\n")
            f.write(f"  Std:    {stats['std']:.4f}\n")
            f.write(f"  Count:  {stats['count']}\n\n")

        f.write("=" * 60 + "\n")

    print(f"Saved summary: {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GTU RIR dataset")
    parser.add_argument('--dataset-path', type=str, default=get_datasets_folder(),
                        help='Path to the dataset pickle file')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data_analysis'),
                        help='Directory to save analysis results')
    parser.add_argument('--debug', type=str2bool, default=False,
                        help='Debug mode: process only 10 RIRs')
    parser.add_argument('--n-abnormal-plots', type=int, default=30,
                        help='Number of secondary peak artifact plots to save')
    parser.add_argument('--sr', type=int, default=44100,
                        help='Sample rate of RIRs (default: 44100)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Determine number of samples
    n_samples = 10 if args.debug else None
    mode_str = "DEBUG (10 samples)" if args.debug else "FULL dataset"
    print(f"\nRunning in {mode_str} mode")
    print(f"Loading dataset from: {args.dataset_path}")

    # Load dataset (without splitting)
    dataset = create_gtu_datasets(
        tar_path=args.dataset_path,
        split=False,
        nSamples=n_samples,
    )
    print(f"Loaded {len(dataset)} samples")

    # Compute metrics
    print("\nComputing metrics...")
    metrics, artifact_results = compute_metrics(
        dataset, sr=args.sr, output_dir=args.output_dir,
        n_abnormal_plots=args.n_abnormal_plots
    )

    # Print artifact detection summary
    print(f"\nSecondary peak artifacts: {artifact_results['count']}/{artifact_results['total']} "
          f"({artifact_results['percentage']:.2f}%)")
    print(f"Artifact plots saved: {artifact_results['n_plots_saved']}")

    # Save histograms
    print("\nSaving histograms...")
    for metric_name, values in metrics.items():
        save_histogram(values, metric_name, args.output_dir,
                       units=METRIC_UNITS.get(metric_name, ''))

    # Save summary
    print("\nSaving summary...")
    save_summary(metrics, artifact_results, args.output_dir, args.debug)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
