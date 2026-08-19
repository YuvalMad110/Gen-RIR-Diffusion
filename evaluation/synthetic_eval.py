#!/usr/bin/env python3
"""
Synthetic RIR Evaluation Script

Evaluates the diffusion model by comparing generated RIRs against synthetic RIRs
generated using either pyroomacoustics or Habets' RIR-Generator (image source method).

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 synthetic_eval.py --n_samples 100
    CUDA_VISIBLE_DEVICES=1 python3 synthetic_eval.py --n_samples 100 --rir_method habets
    CUDA_VISIBLE_DEVICES=2 python3 synthetic_eval.py --model_path /path/to/model.pth.tar --n_samples 50 --guidance_scale 4.0
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from RIRDiffusionModel import RIRDiffusionModel
from utils.signal_proc import spectrogram_to_waveform, undo_rir_scaling, calculate_edc, estimate_decay_k_factor
from utils.inference_data_loading import load_model_and_data_info
from utils.acoustic_metrics import evaluate_rir_pair, aggregate_metrics, align_rir_lengths
from utils.misc import get_israel_time
from utils.evaluation import select_representative_samples
from utils.evaluation_reporting import save_evaluation_summary, save_detailed_metrics_table, save_selected_samples
from utils.visualization import (
    plot_all_histograms, plot_histograms_summary,
    plot_selected_rir_samples, plot_edc_per_band_for_selected
)
from utils.synthetic_rir import generate_synthetic_rirs_batch


# =============================================================================
# Condition Sampling (based on GTU dataset ranges)
# =============================================================================

# GTU dataset ranges (from data exploration)
ROOM_DIM_RANGES = {
    'length': (5.15, 12.30),  # meters
    'width': (5.32, 11.85),
    'height': (2.83, 3.96),
}

RT60_RANGE = (0.45, 2.0)  # seconds

# Margin from walls for mic/speaker placement
WALL_MARGIN = 0.5  # meters


def sample_conditions(n_samples, seed=None):
    """Sample random room conditions within GTU dataset ranges.

    Args:
        n_samples: Number of conditions to sample
        seed: Random seed for reproducibility

    Returns:
        conditions: np.array of shape [n_samples, 10]
                   Format: [room_length, room_width, room_height, mic_x, mic_y, mic_z, speaker_x, speaker_y, speaker_z, rt60]
    """
    if seed is not None:
        np.random.seed(seed)

    conditions = []

    for _ in range(n_samples):
        # Sample room dimensions
        room_length = np.random.uniform(*ROOM_DIM_RANGES['length'])
        room_width = np.random.uniform(*ROOM_DIM_RANGES['width'])
        room_height = np.random.uniform(*ROOM_DIM_RANGES['height'])

        # Sample RT60
        rt60 = np.random.uniform(*RT60_RANGE)

        # Sample mic position (with margin from walls)
        mic_x = np.random.uniform(WALL_MARGIN, room_length - WALL_MARGIN)
        mic_y = np.random.uniform(WALL_MARGIN, room_width - WALL_MARGIN)
        mic_z = np.random.uniform(1.2, min(2.0, room_height - 0.3))  # Typical mic height

        # Sample speaker position (with margin from walls)
        speaker_x = np.random.uniform(WALL_MARGIN, room_length - WALL_MARGIN)
        speaker_y = np.random.uniform(WALL_MARGIN, room_width - WALL_MARGIN)
        speaker_z = np.random.uniform(1.2, min(2.0, room_height - 0.3))  # Typical speaker height

        condition = [room_length, room_width, room_height,
                    mic_x, mic_y, mic_z,
                    speaker_x, speaker_y, speaker_z,
                    rt60]
        conditions.append(condition)

    return np.array(conditions)


# =============================================================================
# Diffusion Model RIR Generation
# =============================================================================

def generate_diffusion_rirs_batch(model, conditions, device, data_info, num_inference_steps, guidance_scale):
    """Generate RIRs using the diffusion model.

    Args:
        model: Trained diffusion model
        conditions: np.array of shape [n_samples, 10]
        device: torch device
        data_info: Data info dict from model checkpoint
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale

    Returns:
        rirs: List of RIR waveforms
    """
    hop_length = data_info['hop_length']
    n_fft = data_info['n_fft']
    sample_size = model.sample_size

    # Convert conditions to tensor
    cond_tensor = torch.tensor(conditions, dtype=torch.float32).to(device)

    batch_size = len(conditions)
    channels = 2  # Real + Imag for spectrogram
    shape = (batch_size, channels, *sample_size)

    print(f"Generating {batch_size} RIRs with diffusion model...")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")

    with torch.no_grad():
        generated_specs = model.generate(
            cond=cond_tensor,
            shape=shape,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            verbose=True
        )

    if torch.is_tensor(generated_specs):
        generated_specs = generated_specs.cpu().numpy()

    # Convert spectrograms to waveforms
    rirs = [spectrogram_to_waveform(generated_specs[i], hop_length, n_fft) for i in range(batch_size)]

    return rirs


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_synthetic(synthetic_rirs, diffusion_rirs, conditions, sr, octave_bands):
    """Evaluate diffusion RIRs against synthetic RIRs.

    Args:
        synthetic_rirs: List of synthetic RIR waveforms (reference)
        diffusion_rirs: List of diffusion-generated RIR waveforms
        conditions: Condition array
        sr: Sample rate
        octave_bands: List of octave band center frequencies

    Returns:
        aggregate: Aggregated metrics
        all_metrics: List of per-sample metrics
        all_samples: List of sample dicts for visualization
    """
    all_metrics = []
    all_samples = []

    print(f"\nEvaluating {len(synthetic_rirs)} RIR pairs...")

    for i in tqdm(range(len(synthetic_rirs)), desc="Computing metrics"):
        rir_syn = synthetic_rirs[i]
        rir_diff = diffusion_rirs[i]

        # Align lengths
        rir_diff, rir_syn = align_rir_lengths(rir_diff, rir_syn, mode='truncate')

        # Compute metrics (synthetic is reference, diffusion is generated)
        metrics = evaluate_rir_pair(rir_diff, rir_syn, sr, octave_bands)
        all_metrics.append(metrics)

        # Store for visualization
        all_samples.append({
            'generated': rir_diff,
            'reference': rir_syn,
            'condition': conditions[i],
            'metrics': metrics
        })

    aggregate = aggregate_metrics(all_metrics)

    return aggregate, all_metrics, all_samples


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RIR Diffusion Model against Synthetic RIRs")
    parser.add_argument("--model_path", type=str, default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/Dec24_20-02-59_dsief07/model_best.pth.tar', help="Path to trained model checkpoint")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="CFG scale (1.0=no guidance)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps (None=use training timesteps)")
    parser.add_argument("--octave_bands", type=float, nargs='+', default=[125, 250, 500, 1000, 2000, 4000], help="Octave bands")
    parser.add_argument("--save_path", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for condition sampling")
    parser.add_argument("--rir_method", type=str, default='habets', choices=['pra', 'habets'],
                        help="Synthetic RIR generation method: 'pra' (pyroomacoustics) or 'habets' (RIR-Generator by Habets)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # ---------- Load model ----------
    print("\nLoading model...")
    model, data_info = load_model_and_data_info(args.model_path, device, RIRDiffusionModel)
    args.num_inference_steps = args.num_inference_steps or model.n_timesteps
    sr = data_info['sr_target']

    # Check CFG
    if args.guidance_scale != 1.0 and not getattr(model, 'guidance_enabled', False):
        print("Warning: Model was not trained with CFG. Setting guidance_scale to 1.0")
        args.guidance_scale = 1.0

    # Output directory
    if args.save_path is None:
        folder_name = f"synthetic_eval_{get_israel_time()}"
        if getattr(model, 'guidance_enabled', False):
            folder_name += f"_guidance{args.guidance_scale}"
        args.save_path = os.path.join(os.path.dirname(args.model_path), folder_name)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # ---------- Sample conditions ----------
    print(f"\nSampling {args.n_samples} random conditions...")
    conditions = sample_conditions(args.n_samples, seed=args.seed)
    print(f"  Room dims range: {conditions[:, :3].min(axis=0)} to {conditions[:, :3].max(axis=0)}")
    print(f"  RT60 range: {conditions[:, 9].min():.2f} to {conditions[:, 9].max():.2f} s")

    # ---------- Generate synthetic RIRs (reference) ----------
    method_label = 'pyroomacoustics' if args.rir_method == 'pra' else 'RIR-Generator (Habets)'
    print(f"\nGenerating synthetic RIRs with {method_label}...")
    # Estimate max RIR length based on longest RT60
    max_rt60 = conditions[:, 9].max()
    max_length_samples = int(max_rt60 * sr * 1.5)  # 1.5x RT60 should capture decay
    synthetic_rirs = generate_synthetic_rirs_batch(conditions, sr, max_length_samples, method=args.rir_method)

    # ---------- Generate diffusion RIRs ----------
    print("\nGenerating RIRs with diffusion model...")
    diffusion_rirs = generate_diffusion_rirs_batch(
        model, conditions, device, data_info,
        args.num_inference_steps, args.guidance_scale
    )

    # ---------- Undo RIR scaling if model was trained with it ----------
    scale_rir = data_info.get('scale_rir', False)
    if scale_rir:
        print("\nUndoing RIR scaling on generated waveforms (model trained with scale_rir=True)...")
        ref_tensor = torch.stack([torch.tensor(w, dtype=torch.float32) for w in synthetic_rirs])
        gen_tensor = torch.stack([torch.tensor(w, dtype=torch.float32) for w in diffusion_rirs])

        edc = calculate_edc(ref_tensor)
        k_factors, _ = estimate_decay_k_factor(edc, sr, data_info.get('db_cutoff', -40))

        gen_unscaled = undo_rir_scaling(gen_tensor, k_factors, sr)
        diffusion_rirs = [g.cpu().numpy() for g in gen_unscaled]

    # ---------- Evaluate ----------
    aggregate, all_metrics, all_samples = evaluate_synthetic(
        synthetic_rirs, diffusion_rirs, conditions, sr, args.octave_bands
    )

    # ---------- Select representative samples ----------
    print("\nSelecting representative samples...")
    metric_names = ['t60_perc']
    selected_samples = select_representative_samples(all_samples, metric_names)

    # ---------- Reporting and Visualization ----------
    print("\nSaving results...")
    save_evaluation_summary(
        aggregate, len(all_metrics), data_info, args,
        args.n_samples, model.n_timesteps,
        save_path / 'evaluation_summary.txt',
        title=f"SYNTHETIC RIR EVALUATION SUMMARY (Diffusion vs {method_label})"
    )
    save_detailed_metrics_table(all_metrics, all_samples, save_path / 'detailed_metrics.csv')
    save_selected_samples(selected_samples, data_info, save_path)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_histograms_summary(all_metrics, len(all_metrics), save_path / 'histograms_summary.png')
    plot_all_histograms(all_metrics, save_path / 'histograms')

    # Plot selected representative samples
    model_name = Path(args.model_path).parent.name
    for metric_name in selected_samples.keys():
        plot_selected_rir_samples(
            selected_samples, metric_name, sr,
            model_name, args.num_inference_steps, model.n_timesteps, save_path, use_rt60_condition=True
        )
        plot_edc_per_band_for_selected(
            selected_samples, metric_name, sr,
            args.octave_bands, save_path
        )

    print(f"\n✓ Synthetic evaluation complete! Results saved to: {save_path}")


if __name__ == "__main__":
    main()
