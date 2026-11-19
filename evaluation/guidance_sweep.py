#!/usr/bin/env python3
"""
Guidance Scale Sweep — evaluate a trained RIR diffusion model across a range of
classifier-free guidance scales and compare results per-RIR.

Iterates the test dataloader once; for each batch, generates RIRs at every scale
and evaluates against the shared reference.

Produces two comparison tables:
  1. Average Score (+rank): mean metric value per scale, ranked.
  2. Average Rank: per-RIR ranking of scales, then averaged.

Usage:
    python guidance_sweep.py --model_path /path/to/model_best.pth.tar
    python guidance_sweep.py --model_path /path/to/model_best.pth.tar --guidance_min 1.0 --guidance_max 5.0 --guidance_step 0.5
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.signal_proc import model_output_to_rir, unscale_rir
from utils.inference_data_loading import (
    load_pretrained_model, data_params_from_run_config,
    build_test_dataloader, _normalize_batch_to_dict,
)
from utils.dataset_utils import build_condition_tensor
from utils.acoustic_metrics import (align_rir_lengths, compute_edc, DEFAULT_OCTAVE_BANDS,
                                    DEFAULT_T60_FIT_RANGE)
from utils.misc import get_israel_time, get_full_path
from evaluation.build_test_references import load_test_references, resolve_split_id
from evaluation.lib.generation_utils import generation_dir, generation_tag, load_generations
from evaluation.lib.metrics_utils import aggregate
from evaluation.lib.result_writers import load_dry_signal, score_rows
from evaluation.pre_generate import resolve_unscaling
from evaluation.run_evaluation import IDENTITY_COLUMNS
import matplotlib.pyplot as plt


# Mapping from metric key to (display_name, extractor, lower_is_better)
# The extractor reads score_pair's flat row, the same row run_evaluation scores into metrics.csv
RANKING_METRICS = {
    't60_perc': ('T60% Error', lambda m: m['t60_perc'], True),
    'edt': ('|EDT| Error (s)', lambda m: m['edt_abs'], True),
    'drr': ('|DRR| Error (dB)', lambda m: m['drr_abs'], True),
    'lsd': ('LSD (dB)', lambda m: m['lsd_min'], True),
    'c50': ('|C50| Error (dB)', lambda m: m['c50_abs'], True),
    'c80': ('|C80| Error (dB)', lambda m: m['c80_abs'], True),
    'edc_distance': ('EDC Dist (dB²)', lambda m: m['edc_distance'], True),
    'cosine_similarity': ('Cosine Sim', lambda m: m['cosine_similarity'], False),
}


# =============================================================================
# Batch processing
# =============================================================================

def prepare_batch(batch, device, src_trgt_dist_cond: bool = False):
    """The conditioning tensor and the sample ids of one dataloader batch.

    Handles both GTU tuple batches and SoundSpaces dict batches. The reference RIRs come from the
    test reference store rather than from here, so the batch's own waveforms are never read.

    Returns:
        (conditions [B, cond_dim] on device, sample ids as a list of str)
    """
    batch = _normalize_batch_to_dict(batch)
    conditions = build_condition_tensor(batch, device, src_trgt_dist_cond)
    return conditions, [str(s) for s in batch['sample_id']]


def generate_for_scale(model, conditions, guidance_scale, sample_size, data_info,
                       num_inference_steps, use_ddim, rt60=None, images=None):
    """Generate RIRs for one guidance scale, un-scaled the way the run was trained.

    The decay is restored from the rt60 the model was conditioned on, matching what
    pre_generate writes, so these waveforms are on the same footing as a stored generation set.

    Returns:
        A list of 1-D numpy arrays, one per sample, in batch order.
    """
    batch_size = conditions.shape[0]
    shape = (batch_size, model.config['in_channels'], *sample_size)

    generated = model.generate(
        cond=conditions, shape=shape, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, verbose=False, use_ddim=use_ddim, images=images)

    waveforms = model_output_to_rir(generated, data_info['rir_domain'], data_info['hop_length'],
                                    data_info['n_fft'], data_info['rir_gain'])        # [B, T]
    if rt60 is not None:
        waveforms = unscale_rir(waveforms.cpu(), data_info['sr_target'], rt60)
    return [w for w in waveforms.cpu().numpy().astype(np.float32)]


def build_tasks_by_id(waveforms: dict, store: dict, store_index: dict, sample_ids: list):
    """Pair each generated RIR with its own reference, matched by sample id.

    Matching by id rather than by row is what lets a sweep score a subset of the split: the
    store holds every test pair, while a sweep may cover only some of them.

    Returns:
        (tasks, identities) in sample_ids order, shaped for score_rows.
    """
    rows = [store_index[sample_id] for sample_id in sample_ids]
    identities = [{column: store[column][row] for column in IDENTITY_COLUMNS} for row in rows]
    tasks = [(waveforms[sample_id], store['references'][row],
              {key: store[key][row] for key in ('rt60', 'c50', 'drr')})
             for sample_id, row in zip(sample_ids, rows)]
    return tasks, identities


def find_reusable_generations(model_dir, scales, args, unscaling: str) -> dict:
    """The scales already on disk as a generation set this sweep can score instead of regenerating.

    A set qualifies when its sampler, step count, seed and un-scaling match what the sweep would
    run. Its waveforms were produced at pre_generate's own batch size, so a reused scale carries a
    different noise draw from a generated one; the summary names which scales were reused.

    Returns:
        {scale: {sample_id: waveform}} for every scale that qualifies.
    """
    sampler = 'ddim' if args.use_ddim else 'ddpm'
    reusable = {}

    for scale in scales:
        tag = generation_tag(sampler, args.num_inference_steps, scale, args.seed)
        target = generation_dir(model_dir, tag)
        if not target.exists():
            continue
        try:
            stored = load_generations(target)
        except Exception as error:
            print(f"  scale {scale}: {target.name} unreadable, regenerating ({error})")
            continue
        if stored['config'].get('unscaling') != unscaling:
            print(f"  scale {scale}: {target.name} un-scaled as "
                  f"{stored['config'].get('unscaling')!r}, regenerating")
            continue
        reusable[scale] = dict(zip((str(s) for s in stored['sample_id']), stored['generated']))

    return reusable


# =============================================================================
# Sample selection for visualization
# =============================================================================

def select_top_scales(avg_score_table, scales, metric_key='t60', n=3):
    """Return the top-n scales based on average score for a given metric."""
    ranks = avg_score_table[metric_key]['ranks']
    sorted_indices = np.argsort(ranks)[:n]
    return [scales[i] for i in sorted_indices]


def select_best_per_scale(results, top_scales, metric_key='t60'):
    """For each top scale, select the sample with the best score (no overlap).

    Returns:
        indices: list of sample indices (one per top scale)
        overlap_info: list of (scale, original_best_idx, used_idx) for cases where overlap was resolved
    """
    _, extractor, lower_is_better = RANKING_METRICS[metric_key]
    used = set()
    indices = []
    overlap_info = []

    for scale in top_scales:
        values = np.array([extractor(m) for m in results[scale]])
        # Sort: ascending for lower_is_better, descending otherwise
        order = np.argsort(values) if lower_is_better else np.argsort(-values)
        original_best = order[0]
        for candidate in order:
            if candidate not in used:
                if candidate != original_best:
                    overlap_info.append((scale, int(original_best), int(candidate)))
                used.add(candidate)
                indices.append(int(candidate))
                break

    return indices, overlap_info


def select_global_medians(results, top_scales, metric_key='t60', n=3, exclude_indices=None):
    """Select n samples that are most consistently 'median' across top scales.

    For each sample, compute sum of |rank - median_rank| across top scales.
    Pick the n samples with the smallest sum (excluding already-used indices).
    """
    _, extractor, lower_is_better = RANKING_METRICS[metric_key]
    exclude = set(exclude_indices or [])
    n_samples = len(results[top_scales[0]])
    median_rank = (n_samples + 1) / 2.0

    # Compute rank per sample per top scale, then sum distance from median
    total_offset = np.zeros(n_samples)
    for scale in top_scales:
        values = np.array([extractor(m) for m in results[scale]])
        ranks = rankdata(values) if lower_is_better else rankdata(-values)
        total_offset += np.abs(ranks - median_rank)

    # Pick n with lowest offset, excluding used indices
    order = np.argsort(total_offset)
    selected = []
    for idx in order:
        if idx not in exclude:
            selected.append(int(idx))
            exclude.add(idx)
            if len(selected) == n:
                break

    return selected


# =============================================================================
# Regeneration and plotting
# =============================================================================

def regenerate_selected(model, test_dataset, sample_indices, top_scales, data_info, device,
                        sample_size, num_inference_steps, use_ddim, collate_fn, unscaling,
                        src_trgt_dist_cond: bool = False):
    """Regenerate waveforms for selected samples at top scales.

    Returns:
        ref_waveforms: dict {idx: 1D numpy array}
        gen_waveforms: dict {(idx, scale): 1D numpy array}
    """
    batch = collate_fn([test_dataset[i] for i in sample_indices])
    conditions, _ = prepare_batch(batch, device, src_trgt_dist_cond)

    batch_dict = _normalize_batch_to_dict(batch)
    images = batch_dict['images'].to(device) if 'images' in batch_dict else None
    rt60 = batch_dict['rt60'].cpu() if unscaling == 'condition_rt60' else None

    rirs = batch_dict['rir']
    ref_waveforms = {idx: rirs[j].cpu().numpy().squeeze() for j, idx in enumerate(sample_indices)}
    gen_waveforms = {}

    with torch.no_grad():
        for scale in top_scales:
            gen_wavs = generate_for_scale(
                model, conditions, scale, sample_size, data_info,
                num_inference_steps, use_ddim, rt60, images=images)
            for j, idx in enumerate(sample_indices):
                gen_waveforms[(idx, scale)] = gen_wavs[j]

    return ref_waveforms, gen_waveforms


def plot_sweep_samples(sample_indices, labels, top_scales, ref_waveforms, gen_waveforms, sr, save_path):
    """Plot waveforms + EDC overlay for selected samples across top scales.

    Layout: len(sample_indices) rows x 5 columns
    Cols: Real waveform | Scale A waveform | Scale B waveform | Scale C waveform | EDC overlay
    """
    n_rows = len(sample_indices)
    n_cols = 1 + len(top_scales) + 1  # Real + scales + EDC
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Colors: green=real, then scale colors
    scale_colors = ['#1f77b4', '#d62728', '#ff7f0e']  # blue, red, orange

    # Column headers
    col_titles = ['Real'] + [f'Scale {s:.1f}' for s in top_scales] + ['EDC Comparison']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight='bold')

    for row, (idx, label) in enumerate(zip(sample_indices, labels)):
        ref = ref_waveforms[idx]
        t_ref = np.arange(len(ref)) / sr * 1000

        # Col 0: Real waveform
        axes[row, 0].plot(t_ref, ref, color='green', linewidth=0.5)
        axes[row, 0].set_ylabel(label, fontsize=8, fontweight='bold')
        axes[row, 0].grid(True, alpha=0.3)

        # EDC overlay — start with real
        edc_col = len(top_scales) + 1
        edc_ref = compute_edc(ref)
        t_edc = np.arange(len(edc_ref)) / sr * 1000
        axes[row, edc_col].plot(t_edc, edc_ref, color='green', linewidth=1.2, label='Real')

        # Cols 1..n: Generated waveforms per scale + EDC overlay
        for k, scale in enumerate(top_scales):
            gen = gen_waveforms[(idx, scale)]
            gen_al, _ = align_rir_lengths(gen, ref, mode='pad')
            t_gen = np.arange(len(gen_al)) / sr * 1000

            axes[row, k + 1].plot(t_gen, gen_al, color=scale_colors[k], linewidth=0.5)
            axes[row, k + 1].grid(True, alpha=0.3)

            edc_gen = compute_edc(gen_al)
            t_edc_gen = np.arange(len(edc_gen)) / sr * 1000
            axes[row, edc_col].plot(t_edc_gen, edc_gen, color=scale_colors[k],
                                    linewidth=1.0, linestyle='--', label=f'{scale:.1f}')

        axes[row, edc_col].legend(fontsize=7, loc='upper right')
        axes[row, edc_col].grid(True, alpha=0.3)
        axes[row, edc_col].set_ylim([-80, 5])
        axes[row, edc_col].set_ylabel('Energy (dB)', fontsize=8)

        # X-axis labels only on last row
        if row == n_rows - 1:
            for j in range(n_cols):
                axes[row, j].set_xlabel('Time (ms)', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sweep sample plot: {save_path}")


# =============================================================================
# Analysis
# =============================================================================

def compute_analysis(results, scales, metric_keys):
    """Compute both comparison tables from per-scale per-sample metrics.

    Args:
        results: dict {scale: [per-sample metric dicts]}
        scales: list of guidance scale values
        metric_keys: list of metric keys to analyze (e.g. ['t60', 'edt', 'drr', 'lsd'])

    Returns:
        avg_score_table: dict {metric_key: {'scores': [float per scale], 'ranks': [int per scale]}}
        avg_rank_table: dict {metric_key: [float mean_rank per scale]}
    """
    n_samples = len(results[scales[0]])
    n_scales = len(scales)

    avg_score_table = {}
    avg_rank_table = {}

    for key in metric_keys:
        _, extractor, lower_is_better = RANKING_METRICS[key]

        # Build scores matrix [n_samples x n_scales]
        scores = np.full((n_samples, n_scales), np.nan)
        for j, scale in enumerate(scales):
            for i, m in enumerate(results[scale]):
                val = extractor(m)
                if val is not None and not np.isnan(val):
                    scores[i, j] = val

        # --- Method 1: Average Score -> Rank ---
        avg_scores = np.nanmean(scores, axis=0)  # [n_scales]
        # Rank the averages (1=best)
        if lower_is_better:
            score_ranks = rankdata(avg_scores, method='average').astype(int)
        else:
            score_ranks = rankdata(-avg_scores, method='average').astype(int)
        avg_score_table[key] = {'scores': avg_scores, 'ranks': score_ranks}

        # --- Method 2: Per-RIR Rank -> Average Rank ---
        # For each sample, rank scales
        rank_matrix = np.full((n_samples, n_scales), np.nan)
        for i in range(n_samples):
            row = scores[i]
            valid = ~np.isnan(row)
            if valid.sum() < 2:
                continue
            if lower_is_better:
                rank_matrix[i, valid] = rankdata(row[valid], method='average')
            else:
                rank_matrix[i, valid] = rankdata(-row[valid], method='average')

        avg_ranks = np.nanmean(rank_matrix, axis=0)  # [n_scales]
        avg_rank_table[key] = avg_ranks

    return avg_score_table, avg_rank_table


def format_summary(scales, metric_keys, avg_score_table, avg_rank_table, n_samples, args,
                   top_scales=None, selected_labels=None, selected_indices=None, overlap_info=None):
    """Format the full sweep summary as a string."""
    lines = []
    lines.append("=" * 80)
    lines.append("GUIDANCE SCALE SWEEP — RESULTS")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {get_israel_time('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Scales: {[f'{s:.1f}' for s in scales]}")
    lines.append(f"Samples evaluated: {n_samples}")
    lines.append(f"Metrics: {metric_keys}")
    lines.append("")

    # --- Table 1: Average Score + Rank ---
    lines.append("-" * 80)
    lines.append("TABLE 1: Average Score per Scale (+ rank)")
    lines.append("-" * 80)

    header = f"{'Scale':>6s}"
    for key in metric_keys:
        name = RANKING_METRICS[key][0]
        header += f" | {name:>14s} (rank)"
    lines.append(header)
    lines.append("-" * len(header))

    for j, scale in enumerate(scales):
        row = f"{scale:6.1f}"
        for key in metric_keys:
            score = avg_score_table[key]['scores'][j]
            rank = avg_score_table[key]['ranks'][j]
            row += f" | {score:14.4f} ({rank:>2d})  "
        lines.append(row)

    lines.append("")

    # --- Table 2: Average Rank ---
    lines.append("-" * 80)
    lines.append("TABLE 2: Average Rank per Scale (per-RIR ranking, then averaged)")
    lines.append("-" * 80)

    header = f"{'Scale':>6s}"
    for key in metric_keys:
        name = RANKING_METRICS[key][0]
        header += f" | {name:>18s}"
    lines.append(header)
    lines.append("-" * len(header))

    for j, scale in enumerate(scales):
        row = f"{scale:6.1f}"
        for key in metric_keys:
            avg_rank = avg_rank_table[key][j]
            row += f" | {avg_rank:18.2f}"
        lines.append(row)

    # --- Visualization Info ---
    if top_scales and selected_labels and selected_indices:
        lines.append("")
        lines.append("=" * 80)
        lines.append("SAMPLE VISUALIZATION")
        lines.append("=" * 80)
        lines.append(f"Top scales (by {args.vis_metric} avg score): {top_scales}")
        for label, idx in zip(selected_labels, selected_indices):
            lines.append(f"  {label}: sample index {idx}")
        if overlap_info:
            lines.append("")
            lines.append("Overlap resolutions (best-per-scale):")
            for sc, orig, used in overlap_info:
                lines.append(f"  Scale {sc:.1f}: original best={orig}, used={used}")

    # --- Config ---
    lines.append("")
    lines.append("=" * 80)
    lines.append("CONFIGURATION")
    lines.append("=" * 80)
    lines.append(f"Model: {args.model_path}")
    lines.append(f"Scheduler: {'DDIM' if args.use_ddim else 'DDPM'}")
    lines.append(f"Inference steps: {args.num_inference_steps}")
    lines.append(f"Batch size: {args.batch_size}")
    lines.append(f"Scale range: {args.guidance_min} to {args.guidance_max} step {args.guidance_step}")
    lines.append(f"Visualization: top {args.n_top_scales} scales, metric={args.vis_metric}, n_medians={args.n_medians}")

    return "\n".join(lines)


def save_csvs(save_path, scales, metric_keys, avg_score_table, avg_rank_table, results):
    """Save ranking tables and full per-sample metrics as CSVs."""
    # Rankings CSV — both methods side by side
    with open(save_path / 'sweep_rankings.csv', 'w') as f:
        # Header
        cols = ['scale']
        for key in metric_keys:
            name = RANKING_METRICS[key][0]
            cols += [f'{name}_avg_score', f'{name}_score_rank', f'{name}_avg_rank']
        f.write(','.join(cols) + '\n')
        for j, scale in enumerate(scales):
            vals = [f'{scale:.1f}']
            for key in metric_keys:
                vals.append(f"{avg_score_table[key]['scores'][j]:.6f}")
                vals.append(f"{avg_score_table[key]['ranks'][j]}")
                vals.append(f"{avg_rank_table[key][j]:.4f}")
            f.write(','.join(vals) + '\n')

    # Full per-sample metrics CSV
    n_samples = len(results[scales[0]])
    with open(save_path / 'sweep_all_metrics.csv', 'w') as f:
        cols = ['sample_idx', 'scale'] + [RANKING_METRICS[k][0] for k in metric_keys]
        f.write(','.join(cols) + '\n')
        for j, scale in enumerate(scales):
            for i, m in enumerate(results[scale]):
                vals = [str(i), f'{scale:.1f}']
                for key in metric_keys:
                    extractor = RANKING_METRICS[key][1]
                    val = extractor(m)
                    vals.append(f"{val:.6f}" if val is not None and not np.isnan(val) else "nan")
                f.write(','.join(vals) + '\n')


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Guidance Scale Sweep Evaluation")
    parser.add_argument("--model_dir", type=str,
                        default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/outputs/finished/Dec24_20-02-59_dsief07',
                        help="Path to run output directory (contains run_config.json + model_best.pth.tar)")
    parser.add_argument("--nSamples", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help='Seed for the sampling noise; also picks which stored generation sets can be reused')
    parser.add_argument("--octave-bands", type=float, nargs='+', default=list(DEFAULT_OCTAVE_BANDS), help='Centre frequencies for the per-band T60')
    parser.add_argument("--t60-fit-range", type=float, nargs=2, default=list(DEFAULT_T60_FIT_RANGE), help='dB window the T60 is fitted over')
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument("--speech_path", type=str, default='/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/test-clean/4992/41797/4992-41797-0013.flac')
    # Sweep-specific
    parser.add_argument("--guidance_min", type=float, default=1.0)
    parser.add_argument("--guidance_max", type=float, default=8.0)
    parser.add_argument("--guidance_step", type=float, default=0.5)
    parser.add_argument("--sweep_metrics", type=str, nargs='+', default=['t60_perc', 'c50', 'drr', 'lsd'],
                        help="Metrics to evaluate and rank. Options: t60_perc, edt, drr, c50, c80, lsd, edc_distance, cosine_similarity")
    # Visualization
    parser.add_argument("--vis_metric", type=str, default='t60_perc',
                        help="Metric used to select top scales and representative samples for plots")
    parser.add_argument("--n_top_scales", type=int, default=3, help="Number of top scales to visualize")
    parser.add_argument("--n_medians", type=int, default=3, help="Number of global-median samples to visualize")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = get_full_path(args.model_dir, 'outputs/finished')
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build list of scales
    scales = list(np.arange(args.guidance_min, args.guidance_max + 1e-9, args.guidance_step))
    scales = [round(s, 2) for s in scales]

    # Validate metrics
    for key in args.sweep_metrics:
        assert key in RANKING_METRICS, f"Unknown metric '{key}'. Options: {list(RANKING_METRICS.keys())}"
    assert args.vis_metric in RANKING_METRICS, f"Unknown vis_metric '{args.vis_metric}'"

    # Both tables compare scales against each other, so one scale leaves nothing to report
    if len(scales) < 2:
        raise SystemExit(
            f"A sweep needs at least two guidance scales; --guidance-min {args.guidance_min} "
            f"--guidance-max {args.guidance_max} --guidance-step {args.guidance_step} gives "
            f"{len(scales)}. Widen the range, or score a single setting with "
            f"pre_generate.py and run_evaluation.py.")

    # Load model
    print("Loading model...")
    model, run_config = load_pretrained_model(model_dir, device)
    data_info = data_params_from_run_config(run_config)
    src_trgt_dist_cond = run_config['model_config'].get('src_trgt_dist_cond', False)
    args.num_inference_steps = args.num_inference_steps or model.n_timesteps
    args.model_path = str(model_dir / 'model_best.pth.tar')  # for summary reporting

    assert getattr(model, 'guidance_enabled', False), \
        "Model was not trained with CFG — guidance sweep requires a CFG-trained model."

    # The LSD's dry speech, loaded the way run_evaluation loads it
    dry_signal = load_dry_signal(args.speech_path, data_info['sr_target'])

    # Debug mode
    if args.debug_mode:
        print("[DEBUG MODE] Overriding settings for fast run")
        args.nSamples = 32
        args.num_inference_steps = 2
        args.batch_size = 16
        scales = scales[:3]

    # Reproducible split
    torch.manual_seed(data_info['random_seed'])
    np.random.seed(data_info['random_seed'])

    # Override nSamples if requested
    if args.nSamples is not None:
        data_info = dict(data_info)
        data_info['nSamples'] = int(args.nSamples / data_info['test_ratio'])

    # Dataset
    test_dataset, test_dataloader = build_test_dataloader(
        data_info, args.batch_size, args.workers, run_config=run_config
    )

    # Output directory
    folder_name = f"guidance_sweep_{get_israel_time()}"
    save_path = model_dir / 'evaluation' / folder_name
    save_path.mkdir(parents=True, exist_ok=True)

    sample_size = tuple(model.sample_size)
    sr = data_info['sr_target']
    unscaling = resolve_unscaling(run_config)

    # The store holds each reference at its native length together with the rt60, C50 and DRR the
    # preparation scripts measured on it, so nothing about the reference is re-measured here
    split_id, _ = resolve_split_id(run_config, test_dataset)
    store = load_test_references(split_id)
    store_index = {str(sample_id): i for i, sample_id in enumerate(store['sample_id'])}

    # ==========================================================================
    # Sweep: one pass over the dataloader, inner loop over scales
    # ==========================================================================
    model.eval()
    reused = find_reusable_generations(model_dir, scales, args, unscaling)
    to_generate = [scale for scale in scales if scale not in reused]
    generated = {scale: dict(waveforms) for scale, waveforms in reused.items()}
    for scale in to_generate:
        generated[scale] = {}

    if reused:
        print(f"Reusing stored generations for scale(s) {sorted(reused)}")

    if to_generate:
        print(f"Generating {len(to_generate)} scale(s) over {len(test_dataset)} samples, "
              f"{len(test_dataloader)} batches...")
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Sweep"):
                conditions, sample_ids = prepare_batch(batch, device, src_trgt_dist_cond)
                batch_dict = _normalize_batch_to_dict(batch)
                images = batch_dict['images'].to(device) if 'images' in batch_dict else None
                rt60 = batch_dict['rt60'].cpu() if unscaling == 'condition_rt60' else None

                for scale in to_generate:
                    waveforms = generate_for_scale(
                        model, conditions, scale, sample_size, data_info,
                        args.num_inference_steps, args.use_ddim, rt60, images=images)
                    generated[scale].update(zip(sample_ids, waveforms))

    # ---------- Score every scale the way run_evaluation does ----------
    print("Scoring...")
    scored_ids = sorted(set.intersection(*(set(generated[scale]) for scale in scales)))
    results = {}
    for scale in scales:
        tasks, identities = build_tasks_by_id(generated[scale], store, store_index, scored_ids)
        results[scale] = score_rows(tasks, identities, sr, dry_signal, args.octave_bands,
                                    tuple(args.t60_fit_range), workers=args.workers)

    n_evaluated = len(scored_ids)

    # ==========================================================================
    # Analysis
    # ==========================================================================
    print("Computing analysis...")
    avg_score_table, avg_rank_table = compute_analysis(results, scales, args.sweep_metrics)

    # ==========================================================================
    # Visualization: select top scales and representative samples
    # ==========================================================================
    print("Selecting samples for visualization...")
    top_scales = select_top_scales(avg_score_table, scales, metric_key=args.vis_metric, n=args.n_top_scales)
    best_indices, overlap_info = select_best_per_scale(results, top_scales, metric_key=args.vis_metric)
    median_indices = select_global_medians(results, top_scales, metric_key=args.vis_metric,
                                           n=args.n_medians, exclude_indices=best_indices)

    all_selected = best_indices + median_indices
    labels = ([f'Best for {s:.1f}' for s in top_scales] +
              [f'Global Median #{i+1}' for i in range(len(median_indices))])

    print("Regenerating waveforms for selected samples...")
    ref_waveforms, gen_waveforms = regenerate_selected(
        model, test_dataset, all_selected, top_scales, data_info, device,
        sample_size, args.num_inference_steps, args.use_ddim, test_dataloader.collate_fn,
        unscaling, src_trgt_dist_cond)

    plot_sweep_samples(all_selected, labels, top_scales, ref_waveforms, gen_waveforms,
                       sr, save_path / 'sweep_samples.png')

    # ==========================================================================
    # Save all outputs
    # ==========================================================================
    summary = format_summary(scales, args.sweep_metrics, avg_score_table, avg_rank_table,
                             n_evaluated, args, top_scales=top_scales,
                             selected_labels=labels, selected_indices=all_selected,
                             overlap_info=overlap_info)
    with open(save_path / 'sweep_summary.txt', 'w') as f:
        f.write(summary + "\n")
    save_csvs(save_path, scales, args.sweep_metrics, avg_score_table, avg_rank_table, results)

    print(f"Done. Results saved to: {save_path}")


if __name__ == "__main__":
    main()
