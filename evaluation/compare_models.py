#!/usr/bin/env python3
"""
Two-model RIR comparison script.

Selects 6 test-set pairs from each model's existing .pt evaluation file,
re-runs DDIM inference through both models for every pair, and generates:
  1. model_comparison.png  — 6 rows × 5 cols (Real | Gen-I | Gen-II | EDC | Specs)
  2. edc_per_band.png      — 6 rows × N_bands per-octave EDC comparison

Row order: Med-I #1, Med-I #2, Med-II #1, Med-II #2, Best-I, Best-II.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 evaluation/compare_models.py \\
        --model_1 Jun08_19-28-08_dsief06 \\
        --model_2 May28_17-36-33_dsief08 \\
        --metric t60_perc \\
        --t60_fit_range -5 -25 \\
        --use_ddim \\
            

    python3 evaluation/compare_models.py   --model_1 Jun08_19-28-08_dsief06   --model_2 May28_17-36-33_dsief08 --use_ddim --font_scale 1.8 --model_names Gen-Vision Gen-Scalar --plot_max_sec 0.4
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.signal_proc import spectrogram_to_waveform, undo_rir_scaling, calculate_edc, estimate_decay_k_factor
from utils.inference_data_loading import (
    load_pretrained_model, data_params_from_run_config,
    build_test_dataloader,
)
from utils.dataset_utils import build_condition_tensor
from utils.acoustic_metrics import evaluate_rir_pair, align_rir_lengths, compute_edc, DEFAULT_T60_FIT_RANGE
from utils.misc import get_israel_time, get_full_path
import soundfile as sf
from utils.audio_processing import load_speech, convolve_with_rir


# =============================================================================
# Sample selection
# =============================================================================

def find_latest_pt(model_dir: Path, metric: str) -> Path:
    """Return path to selected_samples_<metric>.pt in the most recent eval subdir."""
    eval_root = model_dir / 'evaluation'
    if not eval_root.exists():
        raise FileNotFoundError(f"No evaluation/ directory found under {model_dir}")

    candidates = sorted(
        [d for d in eval_root.iterdir()
         if d.is_dir() and (d / f'selected_samples_{metric}.pt').exists()],
        key=lambda d: d.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No eval subdir with selected_samples_{metric}.pt found under {eval_root}"
        )
    pt_path = candidates[-1] / f'selected_samples_{metric}.pt'
    print(f"  Using .pt file: {pt_path}")
    return pt_path


def load_row_indices(pt1: Path, pt2: Path, metric: str) -> List[Tuple[str, int, float]]:
    """Return 6 (label, test_idx, metric_value) tuples in row order."""
    data1 = torch.load(pt1, map_location='cpu', weights_only=False)
    data2 = torch.load(pt2, map_location='cpu', weights_only=False)

    samples1 = data1['samples']
    samples2 = data2['samples']

    rows = [
        ('Med-I #1',  samples1['median'][0]['idx'], samples1['median'][0]['value']),
        ('Med-I #2',  samples1['median'][1]['idx'], samples1['median'][1]['value']),
        ('Med-II #1', samples2['median'][0]['idx'], samples2['median'][0]['value']),
        ('Med-II #2', samples2['median'][1]['idx'], samples2['median'][1]['value']),
        ('Best-I',    samples1['best'][0]['idx'],   samples1['best'][0]['value']),
        ('Best-II',   samples2['best'][0]['idx'],   samples2['best'][0]['value']),
    ]
    print(f"\nSelected test-set indices:")
    for label, idx, val in rows:
        print(f"  {label:10s}  idx={idx:4d}  {metric}={val:.4f}")
    return rows


# =============================================================================
# Per-pair inference
# =============================================================================

def run_inference_for_pair(
    model,
    batch: Dict,
    device: torch.device,
    data_info: Dict,
    args,
    src_trgt_dist_cond: bool,
) -> np.ndarray:
    """Run DDIM inference for a single batch (B=1) and return the generated waveform.

    Applies undo_rir_scaling when scale_rir is active, using k_factor derived from
    the reference RIR in the batch.
    """
    sr = data_info['sr_target']
    hop_length = data_info['hop_length']
    n_fft = data_info['n_fft']
    scale_rir = data_info.get('scale_rir', False)

    conditions = build_condition_tensor(batch, device, src_trgt_dist_cond)

    has_image_encoder = (
        hasattr(model, 'image_encoder') and model.image_encoder is not None
    )
    images = None
    if has_image_encoder and 'images' in batch:
        images = batch['images'].to(device)

    shape = (1, 2, *model.sample_size)
    with torch.no_grad():
        gen_spec = model.generate(
            cond=conditions, shape=shape,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            verbose=False,
            use_ddim=args.use_ddim,
            images=images,
        )

    if torch.is_tensor(gen_spec):
        gen_spec = gen_spec.cpu().numpy()

    gen_wave = spectrogram_to_waveform(gen_spec[0], hop_length, n_fft)

    if scale_rir:
        rirs = batch['rir']
        ref_np = rirs[0].cpu().numpy().squeeze() if torch.is_tensor(rirs[0]) else np.array(rirs[0]).squeeze()

        real_t = torch.tensor(ref_np, dtype=torch.float32).unsqueeze(0)
        edc = calculate_edc(real_t)
        k_factors, _ = estimate_decay_k_factor(edc, sr, data_info.get('db_cutoff', -40))

        gen_t = torch.tensor(gen_wave, dtype=torch.float32).unsqueeze(0)
        gen_wave = undo_rir_scaling(gen_t, k_factors, sr)[0].cpu().numpy()

    return gen_wave


# =============================================================================
# Figures
# =============================================================================

def _edc_db(rir: np.ndarray) -> np.ndarray:
    return compute_edc(rir)


def plot_main_figure(
    rows: List[Dict],
    model_1_name: str,
    model_2_name: str,
    sr: int,
    t60_fit_range: Tuple[float, float],
    save_path: Path,
    model_names: List[str] = None,
    plot_max_ms: float = 500.0,
    font_scale: float = 1.0,
) -> None:
    """6 rows × 5 cols: Real | model_names[0] | model_names[1] | EDC overlay | Specs text box."""
    if model_names is None:
        model_names = ['Gen-I', 'Gen-II']

    # Font sizes derived from a single scale factor
    fs = font_scale
    F_MAIN   = round(14 * fs)
    F_SUB    = round(12 * fs)
    F_COL    = round(10 * fs)
    F_AXIS   = round(9  * fs)
    F_TICK   = round(8  * fs)
    F_LEG    = round(7  * fs)
    F_TEXT   = round(8  * fs)

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 5, figsize=(28, 3.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{model_1_name} ({model_names[0]})  vs  {model_2_name} ({model_names[1]})",
        fontsize=F_MAIN, fontweight='bold', y=1.01,
    )

    for r, row in enumerate(rows):
        label = row['label']
        ref   = row['ref']
        gen1  = row['gen_1']
        gen2  = row['gen_2']
        cond  = row['condition']
        m1    = row['metrics_1']
        m2    = row['metrics_2']

        t_ref  = np.arange(len(ref))  / sr * 1000
        t_gen1 = np.arange(len(gen1)) / sr * 1000
        t_gen2 = np.arange(len(gen2)) / sr * 1000
        x_max  = plot_max_ms  # same clip for waveforms and EDC

        for ax in axes[r]:
            ax.tick_params(labelsize=F_TICK)

        # col 0 — Real
        axes[r, 0].plot(t_ref, ref, color='green', linewidth=0.7)
        axes[r, 0].set_xlim([0, x_max])
        axes[r, 0].set_ylabel('Amplitude', fontsize=F_AXIS)
        axes[r, 0].set_title('Real', fontsize=F_COL, fontweight='bold')
        axes[r, 0].set_xlabel('Time [ms]', fontsize=F_AXIS)
        axes[r, 0].grid(True, alpha=0.3)

        # col 1 — model_names[0]
        axes[r, 1].plot(t_gen1, gen1, color='steelblue', linewidth=0.7)
        axes[r, 1].set_xlim([0, x_max])
        axes[r, 1].set_title(model_names[0], fontsize=F_COL, fontweight='bold')
        axes[r, 1].set_xlabel('Time [ms]', fontsize=F_AXIS)
        axes[r, 1].grid(True, alpha=0.3)

        # col 2 — model_names[1]
        axes[r, 2].plot(t_gen2, gen2, color='darkorange', linewidth=0.7)
        axes[r, 2].set_xlim([0, x_max])
        axes[r, 2].set_title(model_names[1], fontsize=F_COL, fontweight='bold')
        axes[r, 2].set_xlabel('Time [ms]', fontsize=F_AXIS)
        axes[r, 2].grid(True, alpha=0.3)

        # col 3 — EDC overlay (clipped at same x_max)
        edc_ref  = _edc_db(ref)
        edc_gen1 = _edc_db(gen1)
        edc_gen2 = _edc_db(gen2)
        t_edc_ref  = np.arange(len(edc_ref))  / sr * 1000
        t_edc_gen1 = np.arange(len(edc_gen1)) / sr * 1000
        t_edc_gen2 = np.arange(len(edc_gen2)) / sr * 1000

        axes[r, 3].plot(t_edc_ref,  edc_ref,  color='green',      linewidth=1.2, label='Real')
        axes[r, 3].plot(t_edc_gen1, edc_gen1, color='steelblue',  linewidth=1.2, linestyle='--', label=model_names[0])
        axes[r, 3].plot(t_edc_gen2, edc_gen2, color='darkorange', linewidth=1.2, linestyle=':',  label=model_names[1])
        axes[r, 3].axhline(-40, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        axes[r, 3].set_xlim([0, x_max])
        axes[r, 3].set_ylim([-80, 5])
        axes[r, 3].set_title('EDC', fontsize=F_COL, fontweight='bold')
        axes[r, 3].legend(fontsize=F_LEG, loc='upper right')
        axes[r, 3].grid(True, alpha=0.3)
        axes[r, 3].set_ylabel('Energy [dB]', fontsize=F_AXIS)
        axes[r, 3].set_xlabel('Time [ms]', fontsize=F_AXIS)

        # col 4 — Specs text box
        axes[r, 4].axis('off')
        room = cond[:3]
        rt60 = cond[-1] if len(cond) >= 10 else float('nan')
        mic  = cond[3:6]
        spk  = cond[6:9]
        dist = float(np.linalg.norm(spk - mic))

        def _safe(v, fmt):
            return fmt % v if v is not None and not np.isnan(float(v)) else 'N/A'

        txt = (
            f"CONDITION:\n"
            f"  Room: {room[0]:.1f}×{room[1]:.1f}×{room[2]:.1f} m\n"
            f"  RT60: {rt60:.2f} s\n"
            f"  Mic-Src dist: {dist:.2f} m\n"
            f"\n{model_names[0]} METRICS:\n"
            f"  T60%: {_safe(m1['t60']['perc'], '%.1f')}%\n"
            f"  DRR:  {_safe(m1['drr']['error'], '%+.2f')} dB\n"
            f"  LSD:  {_safe(m1['lsd']['broadband'], '%.2f')} dB\n"
            f"\n{model_names[1]} METRICS:\n"
            f"  T60%: {_safe(m2['t60']['perc'], '%.1f')}%\n"
            f"  DRR:  {_safe(m2['drr']['error'], '%+.2f')} dB\n"
            f"  LSD:  {_safe(m2['lsd']['broadband'], '%.2f')} dB\n"
        )
        axes[r, 4].text(
            0.05, 0.5, txt,
            transform=axes[r, 4].transAxes,
            fontsize=F_TEXT, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
        )

    fig.text(
        0.5, -0.01,
        f"T60 fit range: ({t60_fit_range[0]}, {t60_fit_range[1]}) dB  |  "
        f"Waveform / EDC display clipped at {plot_max_ms:.0f} ms",
        ha='center', fontsize=F_TICK, color='gray',
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1], h_pad=3.5 * font_scale)

    # Row separators and subtitles — must be placed after tight_layout
    fig.canvas.draw()
    for r in range(n_rows):
        pos = axes[r, 0].get_position()
        # Subtitle above each row
        fig.text(
            0.5, pos.y1 + 0.012,
            f'── {rows[r]["label"]} ──',
            ha='center', va='bottom',
            fontsize=F_SUB, fontweight='bold', color='#333333',
            transform=fig.transFigure,
        )
        # Separator line below each row (except the last)
        if r < n_rows - 1:
            pos_next = axes[r + 1, 0].get_position()
            y_line = (pos.y0 + pos_next.y1) / 2
            fig.add_artist(plt.Line2D(
                [0.01, 0.99], [y_line, y_line],
                transform=fig.transFigure,
                color='#AAAAAA', linewidth=1.0, linestyle='-',
            ))

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Main figure saved: {save_path}")


def plot_edc_band_figure(
    rows: List[Dict],
    octave_bands: List[float],
    sr: int,
    model_1_name: str,
    model_2_name: str,
    save_path: Path,
    model_names: List[str] = None,
    plot_max_ms: float = 500.0,
    font_scale: float = 1.0,
) -> None:
    """6 rows × N_bands cols: per-octave EDC with 3 traces (Real, model_names[0], model_names[1])."""
    from utils.acoustic_metrics import compute_edc_octave_bands

    if model_names is None:
        model_names = ['Gen-I', 'Gen-II']

    fs = font_scale
    F_MAIN = round(13 * fs)
    F_SUB  = round(11 * fs)
    F_COL  = round(9  * fs)
    F_AXIS = round(8  * fs)
    F_TICK = round(7  * fs)
    F_LEG  = round(7  * fs)

    n_rows = len(rows)
    n_bands = len(octave_bands)
    fig, axes = plt.subplots(n_rows, n_bands, figsize=(3.5 * n_bands, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_bands == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"EDC per Octave Band — {model_1_name} ({model_names[0]})  vs  {model_2_name} ({model_names[1]})",
        fontsize=F_MAIN, fontweight='bold', y=1.01,
    )

    for r, row in enumerate(rows):
        label = row['label']
        ref   = row['ref']
        gen1  = row['gen_1']
        gen2  = row['gen_2']

        edc_ref  = compute_edc_octave_bands(ref,  sr, octave_bands)
        edc_gen1 = compute_edc_octave_bands(gen1, sr, octave_bands)
        edc_gen2 = compute_edc_octave_bands(gen2, sr, octave_bands)

        for b, fc in enumerate(octave_bands):
            ax = axes[r, b]
            if fc not in edc_ref:
                ax.axis('off')
                continue

            edc_r  = edc_ref[fc]
            edc_g1 = edc_gen1.get(fc)
            edc_g2 = edc_gen2.get(fc)

            t_r = np.arange(len(edc_r)) / sr * 1000
            ax.plot(t_r, edc_r, color='green', linewidth=1.0, label='Real')
            if edc_g1 is not None:
                t_g1 = np.arange(len(edc_g1)) / sr * 1000
                ax.plot(t_g1, edc_g1, color='steelblue',  linewidth=1.0, linestyle='--', label=model_names[0])
            if edc_g2 is not None:
                t_g2 = np.arange(len(edc_g2)) / sr * 1000
                ax.plot(t_g2, edc_g2, color='darkorange', linewidth=1.0, linestyle=':',  label=model_names[1])

            ax.set_xlim([0, plot_max_ms])
            ax.set_ylim([-80, 5])
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=F_TICK)
            ax.set_xlabel('Time [ms]', fontsize=F_AXIS)

            if r == 0:
                ax.set_title(f'{int(fc)} Hz', fontsize=F_COL, fontweight='bold')
            if b == 0:
                ax.set_ylabel(f'Energy [dB]', fontsize=F_AXIS)
            else:
                ax.set_yticklabels([])
            if r == 0 and b == n_bands - 1:
                ax.legend(fontsize=F_LEG, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 1], h_pad=3.5 * font_scale)

    # Row subtitles and separators — placed after tight_layout
    fig.canvas.draw()
    for r in range(n_rows):
        pos = axes[r, 0].get_position()
        fig.text(
            0.5, pos.y1 + 0.012,
            f'── {rows[r]["label"]} ──',
            ha='center', va='bottom',
            fontsize=F_SUB, fontweight='bold', color='#333333',
            transform=fig.transFigure,
        )
        if r < n_rows - 1:
            pos_next = axes[r + 1, 0].get_position()
            y_line = (pos.y0 + pos_next.y1) / 2
            fig.add_artist(plt.Line2D(
                [0.01, 0.99], [y_line, y_line],
                transform=fig.transFigure,
                color='#AAAAAA', linewidth=1.0,
            ))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"EDC per-band figure saved: {save_path}")


# =============================================================================
# Reverb speech
# =============================================================================

_DEFAULT_SPEECH = '/home/yuvalmad/Projects/Gen-RIR-Diffusion/data/1195-130164-0010.wav'

def save_reverb_speech(
    rows: List[Dict],
    speech_paths: List[str],
    sr: int,
    out_dir: Path,
    model_names: List[str],
) -> None:
    """Convolve each speech file with real/gen RIRs for the Med-I #1 and Med-II #1 rows.

    Saves 4 WAVs per (row × speech): clean, real, model_names[0], model_names[1].
    Output: out_dir/reverb_speech/{label}_{speech_stem}_{variant}.wav
    """
    reverb_dir = out_dir / 'reverb_speech'
    reverb_dir.mkdir(exist_ok=True)

    target_rows = [r for r in rows if r['label'] in ('Med-I #1', 'Med-II #1')]

    for speech_path in speech_paths:
        speech, _ = load_speech(speech_path, target_sr=sr)
        if speech is None:
            print(f"  Skipping {speech_path} (failed to load)")
            continue
        speech_stem = Path(speech_path).stem

        for row in target_rows:
            label_safe = row['label'].replace(' ', '_').replace('#', '').replace('-', '_')
            prefix = f"{label_safe}_{speech_stem}"

            sf.write(reverb_dir / f"{prefix}_clean.wav", speech, sr)

            real_rev = convolve_with_rir(speech, row['ref'],   normalize_rir=False, normalize_output=True)
            sf.write(reverb_dir / f"{prefix}_real.wav", real_rev, sr)

            gen1_rev = convolve_with_rir(speech, row['gen_1'], normalize_rir=False, normalize_output=True)
            sf.write(reverb_dir / f"{prefix}_{model_names[0]}.wav", gen1_rev, sr)

            gen2_rev = convolve_with_rir(speech, row['gen_2'], normalize_rir=False, normalize_output=True)
            sf.write(reverb_dir / f"{prefix}_{model_names[1]}.wav", gen2_rev, sr)

    print(f"  Reverb speech saved to: {reverb_dir}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compare two RIR diffusion models on selected test pairs")
    parser.add_argument("--model_1", type=str, required=True,
                        help="Model-I folder name or path (under outputs/finished/)")
    parser.add_argument("--model_2", type=str, required=True,
                        help="Model-II folder name or path (under outputs/finished/)")
    parser.add_argument("--metric", type=str, default="t60_perc",
                        help="Metric used to select rows from .pt files (default: t60_perc)")
    parser.add_argument("--t60_fit_range", type=float, nargs=2, default=list(DEFAULT_T60_FIT_RANGE),
                        help="dB window for T60 EDC slope fit: upper lower (e.g. -5 -25)")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="CFG scale (1.0 = no guidance)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="DDIM denoising steps")
    parser.add_argument("--use_ddim", action="store_true",
                        help="Use DDIM sampling (required for fast inference)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cuda:0', 'cpu')")
    parser.add_argument("--octave_bands", type=float, nargs='+',
                        default=[125, 250, 500, 1000, 2000, 4000])
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers per pair (0 recommended for single-item loads)")
    parser.add_argument("--model_names", nargs=2, default=['Gen-I', 'Gen-II'],
                        metavar=('NAME_I', 'NAME_II'),
                        help="Display names for the two models in figure labels (default: Gen-I Gen-II)")
    parser.add_argument("--plot_max_sec", type=float, default=0.5,
                        help="Waveform and EDC x-axis display limit in seconds (default: 0.5)")
    parser.add_argument("--font_scale", type=float, default=1.0,
                        help="Scale factor for all figure font sizes (default: 1.0)")
    parser.add_argument("--speech_paths", nargs='*', default=[_DEFAULT_SPEECH],
                        metavar='WAV',
                        help="Clean speech WAVs to convolve with real/generated RIRs "
                             "(Med-I #1 and Med-II #1 rows only). Pass --speech_paths with no "
                             "arguments to skip. Default: project speech file.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Resolve directories
    model_1_dir = get_full_path(args.model_1, "outputs/finished")
    model_2_dir = get_full_path(args.model_2, "outputs/finished")
    model_1_name = model_1_dir.name
    model_2_name = model_2_dir.name

    print(f"\nModel I:  {model_1_name}")
    print(f"Model II: {model_2_name}")

    # Load models
    print("\nLoading Model I...")
    model_1, run_config_1 = load_pretrained_model(model_1_dir, device)
    print("Loading Model II...")
    model_2, run_config_2 = load_pretrained_model(model_2_dir, device)

    src_trgt_dist_cond_1 = run_config_1['model_config']['src_trgt_dist_cond']
    src_trgt_dist_cond_2 = run_config_2['model_config']['src_trgt_dist_cond']

    data_info = data_params_from_run_config(run_config_1)

    # Split compatibility check
    seed_1 = run_config_1['args'].get('random_seed')
    seed_2 = run_config_2['args'].get('random_seed')
    ratio_1 = run_config_1['args'].get('test_ratio')
    ratio_2 = run_config_2['args'].get('test_ratio')
    if seed_1 != seed_2 or ratio_1 != ratio_2:
        print(
            f"\nWARNING: Models have different split parameters "
            f"(seed: {seed_1} vs {seed_2}, test_ratio: {ratio_1} vs {ratio_2}). "
            f"Test sets may not be identical — comparison may not be fully fair."
        )
    else:
        print(f"\nSplit check OK: seed={seed_1}, test_ratio={ratio_1}")

    # Find .pt files and select row indices
    print(f"\nFinding .pt files for metric '{args.metric}'...")
    pt_1 = find_latest_pt(model_1_dir, args.metric)
    pt_2 = find_latest_pt(model_2_dir, args.metric)
    row_specs = load_row_indices(pt_1, pt_2, args.metric)

    # Build test dataloader using model_1's run_config (loads images if model_1 uses them)
    print("\nBuilding test dataset...")
    test_dataset, test_dataloader = build_test_dataloader(
        data_info, batch_size=1, workers=args.workers, run_config=run_config_1
    )
    print(f"Test set size: {len(test_dataset)}")

    # CFG sanity check
    if args.guidance_scale != 1.0 and not getattr(model_1, 'guidance_enabled', False):
        print("Warning: Model I was not trained with CFG — setting guidance_scale to 1.0")
        args.guidance_scale = 1.0

    collate_fn = test_dataloader.collate_fn

    # Run inference for all 6 rows
    rows = []
    print(f"\nRunning inference for 6 pairs × 2 models ({args.num_inference_steps} DDIM steps each)...")

    model_1.eval()
    model_2.eval()
    sr = data_info['sr_target']

    for label, idx, metric_val in row_specs:
        print(f"\n  [{label}]  test_idx={idx}  ({args.metric}={metric_val:.4f})")

        subset = torch.utils.data.Subset(test_dataset, [idx])
        loader = torch.utils.data.DataLoader(
            subset, batch_size=1, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn,
        )
        batch = next(iter(loader))

        # Reference waveform
        rirs = batch['rir']
        ref_wave = rirs[0].cpu().numpy().squeeze() if torch.is_tensor(rirs[0]) else np.array(rirs[0]).squeeze()
        condition_np = build_condition_tensor(batch, device, src_trgt_dist_cond_1)[0].cpu().numpy()

        print(f"    Inference — Model I...")
        gen_wave_1 = run_inference_for_pair(model_1, batch, device, data_info, args, src_trgt_dist_cond_1)
        print(f"    Inference — Model II...")
        gen_wave_2 = run_inference_for_pair(model_2, batch, device, data_info, args, src_trgt_dist_cond_2)

        # Align lengths
        gen_1, ref_a = align_rir_lengths(gen_wave_1, ref_wave, mode='truncate')
        gen_2, ref_b = align_rir_lengths(gen_wave_2, ref_wave, mode='truncate')

        metrics_1 = evaluate_rir_pair(gen_1, ref_a, sr, args.octave_bands, fit_range=args.t60_fit_range)
        metrics_2 = evaluate_rir_pair(gen_2, ref_b, sr, args.octave_bands, fit_range=args.t60_fit_range)

        print(
            f"    Model I:  T60%={metrics_1['t60']['perc']:.1f}%  DRR={metrics_1['drr']['error']:.2f}dB  LSD={metrics_1['lsd']['broadband']:.2f}dB"
        )
        print(
            f"    Model II: T60%={metrics_2['t60']['perc']:.1f}%  DRR={metrics_2['drr']['error']:.2f}dB  LSD={metrics_2['lsd']['broadband']:.2f}dB"
        )

        rows.append({
            'label':     label,
            'idx':       idx,
            'ref':       ref_a,
            'gen_1':     gen_1,
            'gen_2':     gen_2,
            'condition': condition_np,
            'metrics_1': metrics_1,
            'metrics_2': metrics_2,
        })

    # Summary table
    n0, n1 = args.model_names
    print("\n" + "=" * 72)
    print(f"{'Row':<12} {f'T60%-{n0}':>10} {f'T60%-{n1}':>10} {f'DRR-{n0}':>9} {f'DRR-{n1}':>9} {f'LSD-{n0}':>8} {f'LSD-{n1}':>8}")
    print("-" * 72)
    for row in rows:
        m1, m2 = row['metrics_1'], row['metrics_2']
        print(
            f"{row['label']:<12} "
            f"{m1['t60']['perc']:>8.1f} {m2['t60']['perc']:>8.1f} "
            f"{m1['drr']['error']:>8.2f} {m2['drr']['error']:>8.2f} "
            f"{m1['lsd']['broadband']:>7.2f} {m2['lsd']['broadband']:>7.2f}"
        )
    print("=" * 72)

    # Output directory
    today = get_israel_time('%d-%m-%Y_%H-%M')
    out_dir = model_1_dir / 'evaluation' / 'model_comparison' / today
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to: {out_dir}")

    plot_max_ms = args.plot_max_sec * 1000

    # Generate figures
    plot_main_figure(
        rows, model_1_name, model_2_name, sr,
        tuple(args.t60_fit_range),
        out_dir / 'model_comparison.png',
        model_names=args.model_names,
        plot_max_ms=plot_max_ms,
        font_scale=args.font_scale,
    )
    plot_edc_band_figure(
        rows, args.octave_bands, sr,
        model_1_name, model_2_name,
        out_dir / 'edc_per_band.png',
        model_names=args.model_names,
        plot_max_ms=plot_max_ms,
        font_scale=args.font_scale,
    )

    if args.speech_paths:
        print("\nGenerating reverb speech samples...")
        save_reverb_speech(rows, args.speech_paths, sr, out_dir, args.model_names)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
