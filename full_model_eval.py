#!/usr/bin/env python3
"""
RIR Diffusion Model Evaluation Script

Comprehensive evaluation of generated RIRs against ground truth on the test set.
Computes acoustic metrics (T60, DRR, EDT, C50, LSD, EDC) and generates statistical reports.

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 ./Projects/Gen-RIR-Diffusion/full_model_eval.py --num_inference_steps 50 --use_ddim --nSamples 1000
    python full_model_eval.py --model_path /path/to/model.pth.tar --guidance_scale 4.0 --batch_size 32
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
from data.rir_dataset import load_rir_dataset
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


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RIR Diffusion Model on Test Set")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint",
                        default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/Dec24_20-02-59_dsief07/model_best.pth.tar')
    parser.add_argument("--dataset_path", type=str, default='./datasets/GTU_rir/GTU_RIR.pickle.dat', help="Path to RIR dataset")
    parser.add_argument("--nSamples", type=int, default=None, help="Number of samples (None=use all from data_info)")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="CFG scale (1.0=no guidance)")
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
        args.nSamples = 32
        args.num_inference_steps = 2
        args.batch_size = 16

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
        folder_name = f"evaluation_{get_israel_time()}"
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

        # Plot EDC per band comparison for selected samples
        plot_edc_per_band_for_selected(selected_samples, metric_name, data_info['sr_target'],
                                        args.octave_bands, save_path)

    print(f"\n✓ Evaluation complete! Results saved to: {save_path}")


if __name__ == "__main__":
    main()
