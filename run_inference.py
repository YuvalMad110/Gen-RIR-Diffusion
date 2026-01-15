#!/usr/bin/env python3
"""
RIR Diffusion Model Inference Script

Generate Room Impulse Responses (RIRs) using a trained diffusion model.

Usage:
    python run_inference.py --model_path /path/to/model.pth.tar --nRIR 5
    python run_inference.py --model_path /path/to/model.pth.tar --nRIR 5 --guidance_scale 3.0
    python run_inference.py --model_path /path/to/model.pth.tar --rir_indices 10 20 30
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path

from RIRDiffusionModel import RIRDiffusionModel
from data.rir_dataset import load_rir_dataset

from utils.signal_proc import (evaluate_rir_quality, sisdr_metric, normalize_signals,
    undo_rir_scaling, apply_rir_scaling, spectrogram_to_waveform, waveform_to_spectrogram)
from utils.signal_edc import create_edc_plots_mode2
from utils.inference_data_loading import load_model_and_data_info, load_dataset_conditions
from utils.visualization import plot_comparison, plot_edc_comparison
from utils.audio_processing import process_speech_convolution, save_audio_files
from utils.io_utils import save_inference_results, save_metrics, save_config_summary, create_output_directory


def generate_rirs(model, conditions, device, hop_length, n_fft,
                  num_inference_steps, guidance_scale=1.0):
    """Generate RIRs and convert to waveforms."""
    batch_size = conditions.shape[0]
    channels = 2
    shape = (batch_size, channels, *model.sample_size)

    print(f"Generating {batch_size} RIRs with {num_inference_steps} steps, guidance_scale={guidance_scale}")

    generated_specs = model.generate(cond=conditions, shape=shape, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    
    if torch.is_tensor(generated_specs):
        generated_specs = generated_specs.numpy()
    
    specs_list = []
    waveforms_list = []
    for i in range(batch_size):
        spec = generated_specs[i]
        specs_list.append(spec)
        waveforms_list.append(spectrogram_to_waveform(spec, hop_length, n_fft))
    
    return specs_list, waveforms_list


def prepare_scaled_unscaled_data(real_rirs_wave, generated_rirs_wave, k_factors,
                                  sr, n_fft, hop_length):
    """Prepare scaled real RIRs and unscaled generated RIRs for comparison."""
    real_tensor = torch.stack([torch.tensor(r, dtype=torch.float32) for r in real_rirs_wave])
    gen_tensor = torch.stack([torch.tensor(r, dtype=torch.float32) for r in generated_rirs_wave])

    real_scaled = apply_rir_scaling(real_tensor, k_factors, sr)
    real_scaled_wave = [r.cpu().numpy() for r in real_scaled]
    real_scaled_spec = [waveform_to_spectrogram(r, hop_length, n_fft) for r in real_scaled_wave]

    gen_unscaled = undo_rir_scaling(gen_tensor, k_factors, sr)
    gen_unscaled_wave = [r.cpu().numpy() for r in gen_unscaled]
    gen_unscaled_spec = [waveform_to_spectrogram(r, hop_length, n_fft) for r in gen_unscaled_wave]

    return gen_unscaled_wave, gen_unscaled_spec, real_scaled_wave, real_scaled_spec


def load_selected_rirs(selected_rir_path, device, data_info):
    """Load selected RIR samples from evaluation file.

    Args:
        selected_rir_path: Path to selected_samples_{metric}.pt file
        device: Device to load conditions tensor to
        data_info: Data info dict with hop_length, n_fft, sr_target, etc.

    Returns:
        conditions: Tensor of conditions [N, 10] on device
        real_rirs_wave: List of reference RIR waveforms
        real_rirs_spec: List of reference RIR spectrograms
        quality_labels: List of quality labels ('best', 'worst', 'median', 'mean')
        k_factors: Tensor of k-factors if scaling is enabled, else None
    """
    print(f"\nLoading selected RIRs from: {selected_rir_path}")

    # Load the saved data
    data = torch.load(selected_rir_path, weights_only=False)
    metric_name = data['metric_name']
    samples_dict = data['samples']

    print(f"Loaded samples for metric: {metric_name}")

    # Extract samples in order: best, worst, median, mean
    quality_order = ['best', 'worst', 'median', 'mean']

    conditions_list = []
    real_rirs_wave = []
    quality_labels = []

    for quality in quality_order:
        sample = samples_dict[quality]['sample']
        conditions_list.append(torch.tensor(sample['condition'], dtype=torch.float32))
        real_rirs_wave.append(sample['reference'])
        quality_labels.append(quality)

        print(f"  {quality}: idx={samples_dict[quality]['idx']}, value={samples_dict[quality]['value']:.4f}")

    conditions = torch.stack(conditions_list).to(device)

    # Compute spectrograms from waveforms
    real_rirs_spec = [waveform_to_spectrogram(w, data_info['hop_length'], data_info['n_fft'])
                      for w in real_rirs_wave]

    # Compute k-factors if RIR scaling is enabled
    k_factors = None
    if data_info.get('scale_rir', False):
        from utils.signal_proc import calculate_edc, estimate_decay_k_factor
        real_tensor = torch.stack([torch.tensor(w, dtype=torch.float32) for w in real_rirs_wave])
        edc = calculate_edc(real_tensor)
        k_factors, _ = estimate_decay_k_factor(edc, data_info['sr_target'], data_info.get('db_cutoff', -40))

    return conditions, real_rirs_wave, real_rirs_spec, quality_labels, k_factors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RIRs using trained diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument("--model_path", type=str, default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/Dec16_17-33-21_dsief07/model_best.pth.tar',
                        help="Path to trained model checkpoint (.pth.tar)")
    
    # Generation settings
    parser.add_argument("--nRIR", type=int, default=5, help="Number of RIRs to generate")
    parser.add_argument("--rir_indices", type=int, nargs='+', default=None, help="Specific dataset indices to use (overrides nRIR count)")
    parser.add_argument("--guidance_scale", type=float, default=2, help="CFG scale (1.0=no guidance, >1.0=stronger conditioning)")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Denoising steps (None=use training timesteps)")
    
    # Paths
    parser.add_argument("--dataset_path", type=str, default='./datasets/GTU_rir/GTU_RIR.pickle.dat', help="Path to RIR dataset")
    parser.add_argument("--save_path", type=str, default=None, help="Output directory (None=model directory/generated_rirs)")
    
    # Device & reproducibility
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu). Auto-detect if not specified")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # Output options
    parser.add_argument("--save_audio", type=bool, default=False, help="Save generated RIRs as WAV files")
    parser.add_argument("--save_results", action="store_true", help="Save inference results (.pt file)")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    
    # Speech convolution
    # parser.add_argument("--speech_path", type=str, default='/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/1195/130164/', help="Path to LibriSpeech dataset for speech convolution")
    parser.add_argument("--speech_path", type=str, default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/data', help="Path to LibriSpeech dataset for speech convolution")
    parser.add_argument("--speech_id", type=str, nargs='+', default=['1195-130164-0010','1195-130164-0013'], help="Optional speaker ID to filter LibriSpeech files")
    parser.add_argument("--n_speech_files", type=int, default=2, help="Number of speech files for convolution")
    parser.add_argument("--norm_rir",  type=bool, default=True, help="Normalize RIRs before speech convolution")
    parser.add_argument("--octaves", type=float, nargs='+', default=[125, 250, 500, 1000, 2000, 4000], help="Octave bands for EDC analysis")

    # Load selected RIRs from evaluation
    parser.add_argument("--load_selected_rirs", action="store_true", help="Load RIRs from selected samples file")
    parser.add_argument("--selected_rir_path", type=str, default=None, help="Path to selected RIR samples file (e.g., selected_samples_t60.pt)")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if args.device else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Load model
    model, data_info = load_model_and_data_info(args.model_path, device, RIRDiffusionModel)
    args.num_inference_steps = args.num_inference_steps or model.n_timesteps

    # Check CFG compatibility
    if args.guidance_scale != 1.0:
        if not getattr(model, 'guidance_enabled', False):
            print("Warning: Model was not trained with CFG. Setting guidance_scale to 1.0")
            args.guidance_scale = 1.0

    # Load conditions and RIRs
    if args.load_selected_rirs:
        # Load from selected RIR samples file
        conditions, real_rirs_wave, real_rirs_spec, rir_indices, k_factors = load_selected_rirs(
            args.selected_rir_path, device, data_info
        )

        # Set save path to same folder as selected_rir_path with subfolder "full_generation_{metric}"
        if args.save_path is None:
            # Extract metric name from filename (e.g., "selected_samples_t60.pt" -> "t60")
            filename = os.path.basename(args.selected_rir_path)
            metric_name = filename.replace('selected_samples_', '').replace('.pt', '')
            args.save_path = os.path.join(os.path.dirname(args.selected_rir_path), f"full_generation_{metric_name}")
    else:
        # Load from dataset (original behavior)
        print("\nLoading dataset...")
        _, eval_dataset, _ = load_rir_dataset(
            name='gtu', path=args.dataset_path, split=True, mode='raw',
            hop_length=data_info['hop_length'], n_fft=data_info['n_fft'],
            use_spectrogram=data_info['use_spectrogram'],
            sample_max_sec=data_info['sample_max_sec'],
            nSamples=data_info['nSamples'], sr_target=data_info['sr_target'],
            train_ratio=data_info['train_ratio'], eval_ratio=data_info['eval_ratio'],
            test_ratio=data_info['test_ratio'], random_seed=data_info['random_seed'],
            split_by_room=data_info['split_by_room']
        )

        # Load conditions from dataset
        conditions, real_rirs_wave, real_rirs_spec, rir_indices, k_factors = load_dataset_conditions(
            eval_dataset, args.nRIR, data_info, args.rir_indices
        )
        conditions = conditions.to(device)

        # Set save path to model directory
        if args.save_path is None:
            args.save_path = os.path.join(os.path.dirname(args.model_path), f"generated_rirs")

    # Create output directory
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate RIRs
    print(f"\nGenerating {len(rir_indices)} RIRs...")
    generated_specs, generated_rirs = generate_rirs(
        model, conditions, device, data_info['hop_length'], data_info['n_fft'],
        args.num_inference_steps, args.guidance_scale
    )
    print(f"Generated {len(generated_rirs)} RIR waveforms")

    # Evaluate quality
    metrics = evaluate_rir_quality(generated_rirs, real_rirs_wave, force_same_length=True)

    # Handle scaling if used in training
    if data_info.get('scale_rir', False) and k_factors is not None:
        gen_unscaled_wave, gen_unscaled_spec, real_scaled_wave, real_scaled_spec = \
            prepare_scaled_unscaled_data(
                real_rirs_wave, generated_rirs, k_factors,
                data_info['sr_target'], data_info['n_fft'], data_info['hop_length']
            )

        if not args.no_plots:
            title = f"Scaled | SR:{data_info['sr_target']}Hz, guidance:{args.guidance_scale}"
            plot_comparison(
                real_scaled_wave, generated_rirs, real_scaled_spec, generated_specs,
                conditions.cpu().numpy(), rir_indices, data_info['sr_target'],
                str(save_path / "comparison_scaled.png"), title, metrics
            )
    else:
        gen_unscaled_wave = generated_rirs
        gen_unscaled_spec = generated_specs

    # Plots
    conditions_np = conditions.cpu().numpy()
    title = f"SR:{data_info['sr_target']}Hz | Guidance:{args.guidance_scale}"

    if not args.no_plots:
        plot_comparison(
            real_rirs_wave, gen_unscaled_wave, real_rirs_spec, gen_unscaled_spec,
            conditions_np, rir_indices, data_info['sr_target'],
            str(save_path / "comparison.png"), title, metrics
        )
        create_edc_plots_mode2(
            real_rirs_wave, gen_unscaled_wave, conditions_np, rir_indices,
            data_info['sr_target'], str(save_path / "edc_comparison.png"),
            metrics, args.octaves, title
        )

    # Speech convolution
    if args.speech_path:
        rirs_for_conv = normalize_signals(gen_unscaled_wave) if args.norm_rir else gen_unscaled_wave
        real_for_conv = normalize_signals(real_rirs_wave) if args.norm_rir else real_rirs_wave

        gen_reverb, real_reverb = process_speech_convolution(
            args.speech_path, rirs_for_conv, real_for_conv, str(save_path),
            data_info['sr_target'], args.speech_id, args.n_speech_files,
            normalize_rirs=False
        )

        if gen_reverb and real_reverb:
            metrics['sisdr'] = {}
            metrics['sisdr']['individual'], metrics['sisdr']['total'] = sisdr_metric(
                gen_reverb, real_reverb, force_same_length=True
            )

    # Save outputs
    save_metrics(metrics, rir_indices, str(save_path), conditions_np, args.guidance_scale)
    save_config_summary(model.config, data_info, args, str(save_path))

    if args.save_audio:
        save_audio_files(generated_rirs, str(save_path), data_info['sr_target'], "generated")
        save_audio_files(real_rirs_wave, str(save_path), data_info['sr_target'], "real")
    
    if args.save_results:
        save_inference_results(
            generated_rirs, generated_specs, conditions_np, model.config,
            args.model_path, rir_indices, str(save_path), args.guidance_scale
        )
    
    # Summary
    print("\n" + "=" * 50)
    print("Generation completed!")
    print(f"  Output: {save_path}")
    print(f"  RIRs generated: {len(generated_rirs)}")
    print(f"  Guidance scale: {args.guidance_scale}")
    if 'mse' in metrics:
        print(f"  MSE: {metrics['mse'].get('total', 'N/A'):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
