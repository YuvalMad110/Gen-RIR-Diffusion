#!/usr/bin/env python3
"""
RIR Diffusion Model Evaluation Script

Comprehensive evaluation of generated RIRs against ground truth on the test set.
Computes acoustic metrics (T60, DRR, EDT, C50, LSD, EDC) and generates statistical reports.
Supports both GTU and SoundSpaces+image-conditioned models (auto-detected from run_config.json).

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 evaluation/full_model_eval.py --model_dir outputs/finished/May17_13-36-25_dsief07
    python3 evaluation/full_model_eval.py --model_dir /path/to/run_dir --num_inference_steps 50 --use_ddim
    CUDA_VISIBLE_DEVICES=2 python3 ./Projects/Gen-RIR-Diffusion/evaluation/full_model_eval.py --model_dir /home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/May28_18-24-47_dsief06
"""

import argparse
import os
import sys
import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.signal_proc import spectrogram_to_waveform, undo_rir_scaling, calculate_edc, estimate_decay_k_factor
from utils.inference_data_loading import (
    load_pretrained_model, data_params_from_run_config,
    build_test_dataloader, _normalize_batch_to_dict,
)
from utils.dataset_utils import build_condition_tensor
from utils.acoustic_metrics import evaluate_rir_pair, aggregate_metrics, align_rir_lengths, compute_t60_batch, DEFAULT_T60_FIT_RANGE, DEFAULT_OCTAVE_BANDS
from utils.misc import get_israel_time, get_full_path
from utils.evaluation import select_representative_samples
from utils.evaluation_reporting import save_evaluation_summary, save_detailed_metrics_table, save_selected_samples
from utils.visualization import (
    plot_all_histograms, plot_histograms_summary,
    plot_selected_rir_samples, plot_edc_per_band_for_selected
)
from utils.synthetic_rir import generate_synthetic_rirs_batch
import soundfile as sf
from utils.audio_processing import convolve_with_rir


# =============================================================================
# Evaluation Core
# =============================================================================

def evaluate_test_set(model, test_dataloader, device, data_info, args, src_trgt_dist_cond, dry_signal=None):
    """Run evaluation over the entire test set.

    Handles both GTU tuple batches and SoundSpaces dict batches (with optional images).

    Returns:
        aggregate: Aggregated diffusion metrics
        all_metrics: List of per-sample diffusion metrics
        sample_pairs: List of sample dicts (generated, reference, condition, metrics, and optionally baseline)
        baseline_aggregate: Aggregated baseline metrics (None if baseline disabled)
        baseline_all_metrics: List of per-sample baseline metrics (None if baseline disabled)
    """
    model.eval()

    all_metrics = []
    baseline_all_metrics = [] if args.baseline_method != 'none' else None
    sample_pairs = []

    sr = data_info['sr_target']
    hop_length = data_info['hop_length']
    n_fft = data_info['n_fft']
    sample_size = model.sample_size
    num_inference_steps = args.num_inference_steps
    octave_bands = args.octave_bands
    scale_rir = data_info.get('scale_rir', False)
    use_baseline = args.baseline_method != 'none'

    # Print evaluation configuration
    print(f"\nEvaluating on test set ({len(test_dataloader)} batches)...")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Scheduler: {'DDIM' if args.use_ddim else 'DDPM'}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Octave bands: {octave_bands}")
    if use_baseline:
        baseline_label = 'pyroomacoustics' if args.baseline_method == 'pra' else 'RIR-Generator (Habets)'
        print(f"Baseline: {baseline_label}")

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating batches"):
            batch = _normalize_batch_to_dict(batch)
            rirs = batch['rir']
            batch_size = rirs.shape[0]

            # Convert RIRs to numpy waveforms
            real_waveforms = []
            for i in range(batch_size):
                rir_np = rirs[i].cpu().numpy() if torch.is_tensor(rirs[i]) else rirs[i]
                real_waveforms.append(rir_np.squeeze())

            conditions = build_condition_tensor(batch, device, src_trgt_dist_cond)
            conditions_np = conditions.cpu().numpy()

            # Optional image conditioning
            images = batch['images'].to(device) if 'images' in batch else None

            # --- Generate baseline RIRs (synthetic) ---
            baseline_waveforms = None
            if use_baseline:
                rt60_estimates = None if data_info.get('use_rt60_condition', False) else compute_t60_batch(real_waveforms, sr, args.t60_fit_range)
                baseline_waveforms = generate_synthetic_rirs_batch(
                    conditions_np, sr, max_length_samples=None, method=args.baseline_method, verbose=False, rt60_estimates=rt60_estimates, dataset_name=data_info.get('dataset_name', 'gtu'))

            # --- Generate diffusion RIRs ---
            channels = 2  # Real + Imag for spectrogram
            shape = (batch_size, channels, *sample_size)

            generated_specs = model.generate(
                cond=conditions, shape=shape,
                num_inference_steps=num_inference_steps, guidance_scale=args.guidance_scale, verbose=False,
                use_ddim=args.use_ddim, images=images,
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
                metrics = evaluate_rir_pair(rir_gen, rir_ref, sr, octave_bands, dry_signal=dry_signal, fit_range=args.t60_fit_range)
                all_metrics.append(metrics)

                sample_dict = {
                    'generated': rir_gen,
                    'reference': rir_ref,
                    'condition': conditions_np[i],
                    'metrics': metrics
                }

                # Baseline evaluation
                if use_baseline:
                    rir_base, rir_ref_base = align_rir_lengths(baseline_waveforms[i], real_waveforms[i], mode='truncate')
                    baseline_metrics = evaluate_rir_pair(rir_base, rir_ref_base, sr, octave_bands, dry_signal=dry_signal, fit_range=args.t60_fit_range)
                    baseline_all_metrics.append(baseline_metrics)
                    sample_dict['baseline'] = rir_base
                    sample_dict['baseline_metrics'] = baseline_metrics

                sample_pairs.append(sample_dict)

    aggregate = aggregate_metrics(all_metrics)
    baseline_aggregate = aggregate_metrics(baseline_all_metrics) if use_baseline else None

    return aggregate, all_metrics, sample_pairs, baseline_aggregate, baseline_all_metrics


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RIR Diffusion Model on Test Set")
    parser.add_argument("--model_dir", type=str,
                        default='Dec24_20-02-59_dsief07',
                        help="Run directory or bare folder name under outputs/finished/")
    parser.add_argument("--nSamples", type=int, default=None, help="Override test set size (None=use all)")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="CFG scale (1.0=no guidance)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--octave_bands", type=float, nargs='+', default=DEFAULT_OCTAVE_BANDS)
    parser.add_argument("--save_path", type=str, default=None, help="Output directory")
    parser.add_argument("--n_examples", type=int, default=5, help="Number of example pairs to visualize")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--debug_mode", type=bool, default=False, help="Debug mode: fast run")
    parser.add_argument("--speech_path", type=str, default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/data/1195-130164-0010.wav',
                        help="Path to clean speech for reverbed LSD computation")
    parser.add_argument("--baseline_method", type=str, default='none', choices=['habets', 'pra', 'none'],
                        help="Synthetic RIR baseline method (habets is preferred over pra)")
    parser.add_argument("--t60_fit_range", type=float, nargs=2, default=list(DEFAULT_T60_FIT_RANGE),
                        help="dB window for RT60 EDC slope fit: upper lower (e.g. -5 -35)")
    return parser.parse_args()


def save_reverb_speech(selected_samples: dict, dry_signal: np.ndarray, sr: int, save_path: Path) -> None:
    """Convolve dry_signal with median, best, and worst real/generated RIRs and save 7 WAV files.

    Output: save_path/reverb_speech/{variant}.wav
    Variants: clean, median_real, median_gen, best_real, best_gen, worst_real, worst_gen.
    Uses the first sample from 't60_perc' median, best, and worst categories.
    """
    t60_samples = selected_samples.get('t60_perc', {})
    median_entry = t60_samples.get('median', [{}])[0].get('sample')
    best_entry   = t60_samples.get('best',   [{}])[0].get('sample')
    worst_entry  = t60_samples.get('worst',  [{}])[0].get('sample')

    if median_entry is None or best_entry is None:
        print("  No selected samples available for reverb speech saving.")
        return

    reverb_dir = save_path / 'reverb_speech'
    reverb_dir.mkdir(exist_ok=True)

    sf.write(reverb_dir / 'clean.wav', dry_signal, sr)

    entries = [('median', median_entry), ('best', best_entry)]
    if worst_entry is not None:
        entries.append(('worst', worst_entry))

    for label, sample in entries:
        real_rev = convolve_with_rir(dry_signal, sample['reference'], normalize_rir=False, normalize_output=True)
        gen_rev  = convolve_with_rir(dry_signal, sample['generated'], normalize_rir=False, normalize_output=True)
        sf.write(reverb_dir / f'{label}_real.wav', real_rev, sr)
        sf.write(reverb_dir / f'{label}_gen.wav',  gen_rev,  sr)

    print(f"  Reverb speech saved to: {reverb_dir}")


def main():
    args = parse_args()

    model_dir = get_full_path(args.model_dir, "outputs/finished")

    # Setup device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # ---------- Load model ----------
    print("\nLoading model...")
    model, run_config = load_pretrained_model(model_dir, device)
    data_info = data_params_from_run_config(run_config)
    src_trgt_dist_cond = run_config['model_config']['src_trgt_dist_cond']
    args.num_inference_steps = args.num_inference_steps or model.n_timesteps
    args.model_path = str(model_dir / 'model_best.pth.tar')  # for summary reporting

    # ---------- Load clean speech for reverbed LSD ----------
    print(f"\nLoading clean speech from: {args.speech_path}")
    dry_signal, _ = librosa.load(args.speech_path, sr=data_info['sr_target'])
    print(f"Speech loaded: {len(dry_signal)} samples ({len(dry_signal)/data_info['sr_target']:.2f}s)")

    # Debug mode overrides
    if args.debug_mode:
        print("\n[DEBUG MODE] Overriding settings for fast run!!!")
        args.nSamples = 32
        args.num_inference_steps = 2
        args.batch_size = 16

    # Use seed from training for reproducible test split
    torch.manual_seed(data_info['random_seed'])
    np.random.seed(data_info['random_seed'])

    # Calculate total nSamples from desired test set size
    # User specifies test set size, we back-calculate total: total = test_size / test_ratio
    if args.nSamples is not None:
        data_info = dict(data_info)
        data_info['nSamples'] = int(args.nSamples / data_info['test_ratio'])
        print(f"Test set size requested: {args.nSamples} -> Total dataset size: {data_info['nSamples']}")

    # Check CFG
    if args.guidance_scale != 1.0 and not getattr(model, 'guidance_enabled', False):
        print("Warning: Model was not trained with CFG. Setting guidance_scale to 1.0")
        args.guidance_scale = 1.0

    # Output directory
    if args.save_path is None:
        folder_name = f"evaluation_{get_israel_time()}"
        if getattr(model, 'guidance_enabled', False):
            folder_name += f"_guidance{args.guidance_scale}"
        args.save_path = str(model_dir / 'evaluation' / folder_name)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # ---------- Load test dataset ----------
    # For GTU: yields raw 5-tuple batches (no scaling); for SoundSpaces: yields dict batches with optional images
    print("\nLoading test dataset...")
    test_dataset, test_dataloader = build_test_dataloader(
        data_info, args.batch_size, args.workers, run_config=run_config
    )
    print(f"Test set size: {len(test_dataset)}")

    # ---------- Run evaluation ----------
    aggregate, all_metrics, all_samples, baseline_aggregate, baseline_all_metrics = evaluate_test_set(
        model, test_dataloader, device, data_info, args, src_trgt_dist_cond, dry_signal
    )
    use_baseline = args.baseline_method != 'none'

    # Select representative samples based on hardcoded metrics
    print("\nSelecting representative samples...")
    metric_names = ['t60_perc']
    selected_samples = select_representative_samples(all_samples, metric_names)

    # ---------- Reporting and Visualization ----------
    # Print and save results
    save_evaluation_summary(
        aggregate, len(all_metrics), data_info, args, len(test_dataset), model.n_timesteps,
        save_path / 'evaluation_summary.txt',
        baseline_aggregate=baseline_aggregate,
        baseline_all_metrics=baseline_all_metrics if use_baseline else None,
        baseline_method=args.baseline_method if use_baseline else None,
        all_metrics=all_metrics if use_baseline else None,
        t60_fit_range=args.t60_fit_range,
    )
    save_detailed_metrics_table(all_metrics, all_samples, save_path / 'detailed_metrics.csv')
    save_selected_samples(selected_samples, data_info, save_path)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_histograms_summary(
        all_metrics, len(all_metrics), save_path / 'histograms_summary.png',
        baseline_all_metrics=baseline_all_metrics,
    )
    plot_all_histograms(all_metrics, save_path / 'histograms', baseline_all_metrics=baseline_all_metrics)

    # Plot selected representative samples (4 per metric)
    model_name = model_dir.name
    for metric_name in selected_samples.keys():
        plot_selected_rir_samples(selected_samples, metric_name, data_info['sr_target'],
                                  model_name, args.num_inference_steps, model.n_timesteps, save_path,
                                  use_rt60_condition=data_info['use_rt60_condition'])

        plot_edc_per_band_for_selected(selected_samples, metric_name, data_info['sr_target'],
                                        args.octave_bands, save_path)

    print("\nGenerating reverb speech samples...")
    save_reverb_speech(selected_samples, dry_signal, data_info['sr_target'], save_path)

    print(f"\n✓ Evaluation complete! Results saved to: {save_path}")


if __name__ == "__main__":
    main()
