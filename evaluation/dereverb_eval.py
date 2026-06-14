#!/usr/bin/env python3
"""
Dereverberation-Based Evaluation of RIR Generation Quality
===========================================================

Motivation
----------
A well-trained dereverberation model is a sensitive probe of whether generated
RIRs behave like real measured RIRs.  If the model degrades when given speech
reverbed with *generated* RIRs compared to speech reverbed with *real* RIRs,
it means the generated RIRs are out-of-distribution (OOD) — their acoustic
characteristics differ from the real GTU measurements the model was trained on.

Evaluation protocol
-------------------
For every RIR in the held-out GTU **test split** (rooms never seen during
dereverberation training):

  1. Extract its room/position/RT60 metadata and pass it to the diffusion
     model to generate a *matched* synthetic RIR for the same conditions.
  2. Pick a random LibriSpeech utterance.
  3. Convolve that utterance with both the real and the generated RIR to
     produce two reverberant signals.
  4. Run the dereverberation model on each.
  5. Score PESQ / STOI / SI-SDR vs. the original clean speech at both the
     *reverberant* stage (before model) and *dereverberated* stage (after).

Results are reported side-by-side for real vs. generated RIRs.
A smaller improvement (Δ) on generated RIRs signals OOD behaviour.

Usage
-----
    # nara_wpe (default, no external model required)
    python evaluation/dereverb_eval.py \
        --gtu_path         /home/yuvalmad/datasets/GTU_rir/GTU_RIR.pickle.dat \
        --librispeech_path /dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test \
        --diffusion_model_path Dec24_20-02-59_dsief07 \
        --dereverb_model nara_wpe --wpe_taps 30 --wpe_iterations 15 \
        --guidance_scale 2.5 --use_ddim

    # unet (requires scene_agnostic_dereverberation on disk)
    python evaluation/dereverb_eval.py \
        --diffusion_model_path Dec24_20-02-59_dsief07 \
        --dereverb_model unet \
        --dereverb_model_dir <path/to/mics1_...> \
        --guidance_scale 2.5 --use_ddim

Key arguments
-------------
diffusion_model_path
    Accepts either (a) a full path to the checkpoint .pth.tar file, (b) a full
    path to the run folder (model_best.pth.tar is appended automatically), or
    (c) just the run folder name (e.g. Dec24_20-02-59_dsief07), which is
    expanded to <project>/outputs/finished/<name>/model_best.pth.tar.

dereverb_model
    Which dereverberation model to use as probe metric.
    'unet'     : Scene-agnostic U-Net. Requires scene_agnostic_dereverberation
                 project on disk and --dereverb_model_dir to be set.
    'nara_wpe' : Blind WPE algorithm — no external project needed (default).
"""

import argparse
import pickle
import sys
from datetime import datetime
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve, resample_poly, correlate
from tqdm import tqdm
from pesq import pesq as compute_pesq
from pystoi import stoi as compute_stoi


# =============================================================================
# Path setup — make Gen-RIR-Diffusion importable
# =============================================================================

# This script lives at Gen-RIR-Diffusion/evaluation/dereverb_eval.py
GEN_RIR_ROOT = Path(__file__).parents[1]   # .../Gen-RIR-Diffusion/
EVAL_DIR     = Path(__file__).parent       # .../Gen-RIR-Diffusion/evaluation/

if str(GEN_RIR_ROOT) not in sys.path:
    sys.path.insert(0, str(GEN_RIR_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(1, str(EVAL_DIR))

# --- Gen-RIR-Diffusion imports -----------------------------------------------
from utils.inference_data_loading import (
    load_pretrained_model, data_params_from_run_config,
    build_test_dataloader, _normalize_batch_to_dict, build_condition_tensor,
)
from utils.misc import str2bool, resolve_model_dir
from utils.signal_proc import (
    spectrogram_to_waveform,
    undo_rir_scaling,
    calculate_edc,
    estimate_decay_k_factor,
)

# --- Dereverberation model interface -----------------------------------------
from dereverberation_models import build_dereverb_model, DEREVERB_SR, DereverbModel


# =============================================================================
# SECTION 1 — Model loading
# =============================================================================

def load_diffusion_model(model_dir: Path, device: torch.device):
    """Load the RIR generation diffusion model from a run directory.

    Returns
    -------
    model     : RIRDiffusionModel ready for inference
    data_info : dict with training hyper-parameters (sr_target, hop_length,
                n_fft, split ratios, scale_rir flag, …)
    run_config: Full run_config dict (run_config['args'] holds all data params)
    """
    print(f"  Loading diffusion model from:\n    {model_dir}")
    model, run_config = load_pretrained_model(model_dir, device)
    data_info = data_params_from_run_config(run_config)
    print(f"  Native SR : {data_info['sr_target']} Hz | "
          f"scale_rir : {data_info.get('scale_rir', False)}")
    return model, data_info, run_config




# =============================================================================
# SECTION 2 — Data helpers
# =============================================================================

def load_speech_files(librispeech_dir: str) -> list:
    """
    Recursively collect all .wav files from a LibriSpeech directory.
    Works with the standard speaker/chapter/utterance layout.
    """
    files = sorted(Path(librispeech_dir).rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found in {librispeech_dir}")
    print(f"  Found {len(files)} speech files")
    return files


def resample_rir(rir: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Polyphase resample a RIR waveform from *from_sr* to *to_sr*."""
    if from_sr == to_sr:
        return rir
    g = gcd(from_sr, to_sr)
    return resample_poly(rir, to_sr // g, from_sr // g).astype(np.float32)


def make_reverberant(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    Convolve a clean speech signal with a RIR using the same approach as
    the dereverberation model's training pipeline:
      • full convolution, truncated to the original speech length
      • RMS-normalised to target RMS = 0.1
    """
    reverb = fftconvolve(clean, rir, mode="full")[: len(clean)]
    rms = np.sqrt(np.mean(reverb ** 2))
    return reverb / (rms + 1e-8) * 0.1




# =============================================================================
# SECTION 4 — Speech quality metrics
# =============================================================================

def si_sdr(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    L = min(len(clean), len(estimate))
    c = clean[:L]    - clean[:L].mean()
    e = estimate[:L] - estimate[:L].mean()
    dot      = np.dot(c, e)
    s_target = dot / (np.dot(c, c) + 1e-8) * c
    e_noise  = e - s_target
    return float(10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8)))


def _score_pair(ref: np.ndarray, est: np.ndarray) -> dict:
    """
    Align *est* to *ref* via cross-correlation, then return PESQ / STOI / SI-SDR.
    Both signals are cropped (no zero-padding) to avoid tail artifacts.
    """
    xcorr = correlate(est, ref, mode="full")
    lag = int(np.argmax(xcorr)) - (len(ref) - 1)   # positive → est is delayed

    if lag > 0:
        est = est[lag:]   # shift est back; min() below trims ref's tail
    elif lag < 0:
        ref = ref[-lag:]  # shift ref back; min() below trims est's tail

    L = min(len(ref), len(est))
    r, e = ref[:L], est[:L]
    return {
        "pesq":  float(compute_pesq(DEREVERB_SR, r, e, mode="wb")),
        "stoi":  float(compute_stoi(r, e, DEREVERB_SR, extended=False)),
        "sisdr": si_sdr(r, e),
    }


def compute_metrics(clean: np.ndarray, reverb: np.ndarray, dereverb: np.ndarray) -> dict:
    """
    Compute PESQ / STOI / SI-SDR at the reverberant and dereverberated stages.
    Each signal pair is independently aligned to the clean reference via
    cross-correlation before scoring.

    Returns a dict with six entries:
        pesq_reverb / stoi_reverb / sisdr_reverb
        pesq_dereverb / stoi_dereverb / sisdr_dereverb
    """
    rev = _score_pair(clean, reverb)
    der = _score_pair(clean, dereverb)
    return {
        "pesq_reverb":    rev["pesq"],
        "stoi_reverb":    rev["stoi"],
        "sisdr_reverb":   rev["sisdr"],
        "pesq_dereverb":  der["pesq"],
        "stoi_dereverb":  der["stoi"],
        "sisdr_dereverb": der["sisdr"],
    }


# =============================================================================
# SECTION 5 — Batched RIR generation with the diffusion model
# =============================================================================

def generate_rir_batch(model, conditions: torch.Tensor, data_info: dict, device: torch.device,
                       guidance_scale: float, use_ddim: bool, num_inference_steps: int,
                       images=None) -> list:
    """Generate a batch of RIR waveforms conditioned on room/position metadata.

    The diffusion model outputs spectrograms in the model's native sample rate
    (data_info['sr_target']).  This function converts them to waveforms and
    returns them still at that native rate; resampling to DEREVERB_SR happens
    in the evaluation loop after any necessary amplitude un-scaling.

    Returns
    -------
    list of 1-D numpy float32 arrays at data_info['sr_target']
    """
    batch_size = conditions.shape[0]
    channels   = 2   # real + imaginary parts of the spectrogram
    shape      = (batch_size, channels, *model.sample_size)

    gen_specs = model.generate(cond=conditions, shape=shape, num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale, verbose=False, use_ddim=use_ddim,
                               images=images)

    if torch.is_tensor(gen_specs):
        gen_specs = gen_specs.cpu().numpy()

    # spectrogram_to_waveform returns a tensor; convert to numpy here so the
    # rest of the pipeline (resample_rir, fftconvolve) works uniformly.
    return [
        spectrogram_to_waveform(gen_specs[i], data_info["hop_length"], data_info["n_fft"])
        .cpu().numpy().astype(np.float32)
        for i in range(batch_size)
    ]


# =============================================================================
# SECTION 6 — Main evaluation loop
# =============================================================================

def run_evaluation(model, data_info: dict, dereverb_model: DereverbModel, test_dataloader, speech_files: list, device: torch.device, guidance_scale: float, use_ddim: bool, num_inference_steps: int, rng: np.random.Generator) -> tuple:
    """
    Iterate over all batches in the GTU test set.

    For each batch
    ──────────────
    1. Build the [B, 10] condition tensor from room/position/RT60 metadata.
    2. Generate a batch of RIRs with the diffusion model (same conditions as
       the real ones so the comparison is fair).
    3. Undo the amplitude scaling applied during training if scale_rir=True.
    4. Resample both real and generated RIRs to DEREVERB_SR (16 kHz).
    5. For each sample in the batch:
         a. Pick a random LibriSpeech utterance.
         b. Create two reverberant signals (real RIR / generated RIR).
         c. Dereverberate both with the U-Net.
         d. Compute PESQ / STOI / SI-SDR for each pair vs. the clean signal.

    Returns
    -------
    all_real : list of metric dicts for real-RIR reverberant speech
    all_gen  : list of metric dicts for generated-RIR reverberant speech
    """
    native_sr = data_info["sr_target"]
    scale_rir  = data_info.get("scale_rir", False)

    # Minimum clean speech length required by the dereverberation model
    min_speech_samples = dereverb_model.min_speech_samples

    all_real, all_gen, samples = [], [], []

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = _normalize_batch_to_dict(batch)
        rirs = batch['rir']
        batch_size = rirs.shape[0]

        conditions = build_condition_tensor(batch, device)

        # --- Real RIR waveforms at the model's native sample rate ---------------
        real_waves_native = [rirs[i].cpu().numpy().squeeze() for i in range(batch_size)]

        # --- Optional image conditioning ----------------------------------------
        images = batch['images'].to(device) if 'images' in batch else None

        # --- Compute k-factors for amplitude un-scaling (if used in training) ---
        if scale_rir:
            real_tensor = torch.stack(
                [torch.tensor(w, dtype=torch.float32) for w in real_waves_native]
            )
            edc = calculate_edc(real_tensor)
            k_factors, _ = estimate_decay_k_factor(edc, native_sr, data_info.get("db_cutoff", -40))
        else:
            k_factors = None

        # --- Generate RIRs conditioned on the same metadata as the real ones ----
        gen_waves_native = generate_rir_batch(model, conditions, data_info, device,
                                              guidance_scale, use_ddim, num_inference_steps,
                                              images=images)

        # --- Un-scale generated RIRs if amplitude scaling was used in training --
        if scale_rir and k_factors is not None:
            gen_tensor   = torch.stack(
                [torch.tensor(w, dtype=torch.float32) for w in gen_waves_native]
            )
            gen_unscaled = undo_rir_scaling(gen_tensor, k_factors, native_sr)
            gen_waves_native = [g.cpu().numpy() for g in gen_unscaled]

        # --- Resample both RIR sets to 16 kHz for the dereverberation model ----
        real_waves_16k = [resample_rir(w, native_sr, DEREVERB_SR) for w in real_waves_native]
        gen_waves_16k  = [resample_rir(w, native_sr, DEREVERB_SR) for w in gen_waves_native]

        # --- Per-sample: pick speech → convolve → dereverberate → score --------
        for i in range(batch_size):
            # Pick a random LibriSpeech utterance
            speech_path = speech_files[int(rng.integers(len(speech_files)))]
            clean, _ = sf.read(str(speech_path), dtype="float32")
            if clean.ndim > 1:
                clean = clean[:, 0]                 # take left channel if stereo

            # Normalise clean speech (matches dereverberation training pipeline)
            clean = clean / 1.1 / (np.max(np.abs(clean)) + 1e-8)

            # Skip utterances too short for the dereverberation model's STFT
            if len(clean) < min_speech_samples:
                continue

            # Reverberant signals (real and generated RIR)
            reverb_real = make_reverberant(clean, real_waves_16k[i])
            reverb_gen  = make_reverberant(clean, gen_waves_16k[i])

            # Dereverberate
            try:
                dereverb_real = dereverb_model.dereverberate(reverb_real)
                dereverb_gen  = dereverb_model.dereverberate(reverb_gen)
            except RuntimeError:
                continue     # signal too short after STFT — skip

            # Score both pairs
            try:
                all_real.append(compute_metrics(clean, reverb_real, dereverb_real))
                all_gen.append(compute_metrics(clean, reverb_gen,  dereverb_gen))
                if len(samples) < 2:
                    samples.append({"clean": clean.copy(), "reverb_real": reverb_real.copy(), "reverb_gen": reverb_gen.copy(), "dereverb_real": dereverb_real.copy(), "dereverb_gen": dereverb_gen.copy()})
            except Exception:
                continue     # PESQ / STOI can fail on degenerate signals

    return all_real, all_gen, samples


# =============================================================================
# SECTION 7 — Aggregation and reporting
# =============================================================================

def aggregate(metrics_list: list) -> dict:
    """Compute mean ± std for every metric key across all processed samples."""
    keys = metrics_list[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in metrics_list])),
            "std":  float(np.std( [m[k] for m in metrics_list])),
        }
        for k in keys
    }


def print_results_table(real_agg: dict, gen_agg: dict, n_samples: int) -> None:
    """
    Print a human-readable comparison table.

    Columns: metric label | real RIR (mean ± std) | generated RIR (mean ± std) | Δ
    A negative Δ on the dereverberated rows means the model performs worse on
    generated RIRs, suggesting OOD behaviour.
    """
    sep = "=" * 76
    print(f"\n{sep}")
    print("  DEREVERBERATION EVALUATION — Real RIR  vs.  Generated RIR")
    print(f"  n_samples = {n_samples}")
    print(sep)
    print(f"  {'Metric':<22}  {'Real RIR':>18}  {'Generated RIR':>18}  {'Δ (gen−real)':>13}")
    print("-" * 76)

    metric_labels = {
        "pesq_reverb":    "PESQ  [reverberant]",
        "stoi_reverb":    "STOI  [reverberant]",
        "sisdr_reverb":   "SI-SDR[reverberant]",
        "pesq_dereverb":  "PESQ  [dereverberated]",
        "stoi_dereverb":  "STOI  [dereverberated]",
        "sisdr_dereverb": "SI-SDR[dereverberated]",
    }

    for key, label in metric_labels.items():
        r = real_agg[key]
        g = gen_agg[key]
        delta = g["mean"] - r["mean"]
        print(
            f"  {label:<22}  {r['mean']:>7.3f}±{r['std']:.3f}   "
            f"{g['mean']:>7.3f}±{g['std']:.3f}   {delta:>+9.3f}"
        )
        if key == "sisdr_reverb":
            print()   # blank line separating reverberant / dereverberated blocks

    print(sep)
    print()


def _save_tables(real_agg: dict, gen_agg: dict, args, n_samples: int, save_path: Path) -> None:
    """Write a plain-text file with three comparison tables."""
    metrics = [("pesq", "PESQ"), ("stoi", "STOI"), ("sisdr", "SI-SDR")]

    def imp(agg, key):
        return agg[f"{key}_dereverb"]["mean"] - agg[f"{key}_reverb"]["mean"]

    W = 75
    lines = []
    lines += [f"Dereverberation Evaluation Results  |  n_samples={n_samples}  |  {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * W, ""]

    for label, agg in [("Real RIR", real_agg), ("Generated RIR", gen_agg)]:
        lines += [label, "-" * W]
        lines += [f"  {'Metric':<8}  {'Reverberant (mean±std)':>24}  {'Dereverberated (mean±std)':>26}  {'Improvement Δ':>14}"]
        lines += [f"  {'-'*8}  {'-'*24}  {'-'*26}  {'-'*14}"]
        for key, name in metrics:
            r = agg[f"{key}_reverb"]
            d = agg[f"{key}_dereverb"]
            lines.append(f"  {name:<8}  {r['mean']:>8.3f} ± {r['std']:.3f}        {d['mean']:>8.3f} ± {d['std']:.3f}        {imp(agg, key):>+10.3f}")
        lines.append("")

    lines += ["Comparison: Generated − Real", "-" * W]
    lines += [f"  {'Metric':<8}  {'Reverberant Δ':>14}  {'Dereverberated Δ':>18}  {'Improvement Δ':>14}"]
    lines += [f"  {'-'*8}  {'-'*14}  {'-'*18}  {'-'*14}"]
    for key, name in metrics:
        d_rev = gen_agg[f"{key}_reverb"]["mean"]  - real_agg[f"{key}_reverb"]["mean"]
        d_der = gen_agg[f"{key}_dereverb"]["mean"] - real_agg[f"{key}_dereverb"]["mean"]
        d_imp = imp(gen_agg, key) - imp(real_agg, key)
        lines.append(f"  {name:<8}  {d_rev:>+14.3f}  {d_der:>+18.3f}  {d_imp:>+14.3f}")
    lines.append("")

    lines += ["Config", "-" * W,
              f"  diffusion_model  : {args.diffusion_model_path}",
              f"  dereverb_model   : {args.dereverb_model}",
              f"  guidance_scale   : {args.guidance_scale}",
              f"  steps            : {args.num_inference_steps}",
              f"  ddim             : {args.use_ddim}"]

    out_file = save_path / "dereverb_eval_results.txt"
    out_file.write_text("\n".join(lines))
    print(f"  Tables saved to  : {out_file}")


def _save_boxplots(all_real: list, all_gen: list, save_path: Path) -> None:
    """Save a grouped boxplot figure: Real vs Generated, Reverberant vs Dereverberated."""
    metrics = [("pesq", "PESQ"), ("stoi", "STOI"), ("sisdr", "SI-SDR (dB)")]
    col_rev  = "#5b9bd5"   # blue   — reverberant
    col_der  = "#ed7d31"   # orange — dereverberated

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Dereverberation Evaluation: Real vs Generated RIRs", fontsize=13, fontweight="bold")

    w = 0.3
    positions = [0.75, 1.25, 1.75, 2.25]   # [real_rev, real_der, gen_rev, gen_der]
    colors    = [col_rev, col_der, col_rev, col_der]

    for ax, (key, name) in zip(axes, metrics):
        data = [
            [m[f"{key}_reverb"]  for m in all_real],
            [m[f"{key}_dereverb"] for m in all_real],
            [m[f"{key}_reverb"]  for m in all_gen],
            [m[f"{key}_dereverb"] for m in all_gen],
        ]
        for pos, d, c in zip(positions, data, colors):
            bp = ax.boxplot(d, positions=[pos], widths=w, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2),
                            flierprops=dict(marker="o", markersize=3, alpha=0.5))
            bp["boxes"][0].set_facecolor(c)
            bp["boxes"][0].set_alpha(0.8)

        ax.set_xlim(0.4, 2.6)
        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels(["Real", "Generated"], fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.axvline(1.5, color="gray", linestyle=":", linewidth=0.8)

    legend_handles = [Patch(facecolor=col_rev, alpha=0.8, label="Reverberant"), Patch(facecolor=col_der, alpha=0.8, label="Dereverberated")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10, frameon=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_file = save_path / "dereverb_eval_boxplots.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Boxplots saved to: {out_file}")


def _save_samples(samples: list, save_path: Path) -> None:
    """Save audio sample sets to samples/sample_NN/ subdirectories."""
    for idx, s in enumerate(samples, 1):
        out_dir = save_path / "samples" / f"sample_{idx:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_dir / "clean.wav"),         s["clean"],         DEREVERB_SR)
        sf.write(str(out_dir / "reverb_real.wav"),   s["reverb_real"],   DEREVERB_SR)
        sf.write(str(out_dir / "reverb_gen.wav"),    s["reverb_gen"],    DEREVERB_SR)
        sf.write(str(out_dir / "dereverb_real.wav"), s["dereverb_real"], DEREVERB_SR)
        sf.write(str(out_dir / "dereverb_gen.wav"),  s["dereverb_gen"],  DEREVERB_SR)
    print(f"  Audio samples saved to: {save_path / 'samples'}")


def save_results(real_agg: dict, gen_agg: dict, all_real: list, all_gen: list, samples: list, args, save_path: Path) -> None:
    _save_tables(real_agg, gen_agg, args, len(all_real), save_path)
    _save_boxplots(all_real, all_gen, save_path)
    _save_samples(samples, save_path)


# =============================================================================
# SECTION 8 — Argument parsing and entry point
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RIR generation quality via dereverberation scores. "
            "Compares PESQ / STOI / SI-SDR on speech reverbed with "
            "real GTU RIRs vs. RIRs generated by Gen-RIR-Diffusion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data paths -----------------------------------------------------------
    parser.add_argument("--librispeech_path", type=str, default="/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test", help="LibriSpeech test directory (will be searched recursively for .wav files)")

    # -- Model paths ----------------------------------------------------------
    parser.add_argument("--diffusion_model_path", type=str, default="Dec24_20-02-59_dsief07", help="Run folder name (e.g. Dec24_20-02-59_dsief07), folder path, or full path to model_best.pth.tar (see script header)")

    # -- Dereverberation model selection --------------------------------------
    parser.add_argument("--dereverb_model", type=str, default="nara_wpe", choices=["unet", "nara_wpe"],
                        help="Dereverberation model to use as probe. 'unet' requires scene_agnostic_dereverberation on disk.")
    # scene agnostic unet settings (only used if --dereverb_model unet)
    parser.add_argument("--dereverb_model_dir", type=str, default=None, help="[unet only] Directory with settings.txt + checkpoints/ sub-folder")
    # nara_wpe settings (only used if --dereverb_model nara_wpe)``
    parser.add_argument("--wpe_taps", type=int, default=30, help="[nara_wpe] WPE filter taps (default: 30)")
    parser.add_argument("--wpe_delay", type=int, default=3, help="[nara_wpe] WPE delay in frames (default: 3)")
    parser.add_argument("--wpe_iterations", type=int, default=15, help="[nara_wpe] WPE refinement iterations (default: 15)")

    # -- Generation settings --------------------------------------------------
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="CFG guidance scale for the diffusion model (1.0 = no guidance)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampler (faster than DDPM, same quality)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for RIR generation (larger = faster but more GPU memory)")

    # -- Evaluation settings --------------------------------------------------
    parser.add_argument("--n_samples", type=int, default=None, help="Number of test RIRs to evaluate (None = full test set)")
    parser.add_argument("--device", type=str, default=None, help='Torch device, e.g. "cuda:0".  Auto-detected if not set.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_mode", type=str2bool, default=False, help="Debug mode: small n_samples/steps/batch for a fast sanity-check run")

    # -- Output ---------------------------------------------------------------
    parser.add_argument("--save_path", type=str, default=None, help="Output directory (default: <diffusion_model_dir>/evaluation/dereverb_eval_<timestamp>/)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Resolve model directory ───────────────────────────────────────────────
    model_dir = resolve_model_dir(args.diffusion_model_path)

    # ── Debug mode ────────────────────────────────────────────────────────────
    if args.debug_mode:
        print("\n[DEBUG MODE] Overriding settings for fast run!!!")
        args.n_samples = 8
        args.num_inference_steps = 2
        args.batch_size = 4

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ── Load models ───────────────────────────────────────────────────────────
    print("── Loading RIR diffusion model ─────────────────────────────────────────")
    model, data_info, run_config = load_diffusion_model(model_dir, device)
    args.num_inference_steps = args.num_inference_steps or model.n_timesteps

    # Warn if guidance is requested but model was not trained with CFG
    if args.guidance_scale != 1.0 and not getattr(model, "guidance_enabled", False):
        print("  Warning: model was not trained with CFG — forcing guidance_scale=1.0")
        args.guidance_scale = 1.0

    print("\n── Loading dereverberation model ───────────────────────────────────────")
    dereverb_model = build_dereverb_model(args, device)

    # ── Output directory (deferred until after models load successfully) ───────
    if args.save_path is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"dereverb_eval_{ts}_guidance{args.guidance_scale}_{dereverb_model.name}"
        if args.debug_mode:
            folder_name += "_DEBUG"
        args.save_path = str(model_dir / 'evaluation' / folder_name)
    save_path = Path(args.save_path)
    print(f"Output : {save_path}\n")

    # ── Load test dataset ─────────────────────────────────────────────────────
    # Reproduce the exact held-out test set used during training by reusing the
    # same random seed and split ratios stored in run_config.
    print("\n── Loading test split ──────────────────────────────────────────────────")
    if args.n_samples is not None:
        data_info = dict(data_info)
        data_info['nSamples'] = int(args.n_samples / data_info["test_ratio"])
    test_dataset, test_dataloader = build_test_dataloader(
        data_info, args.batch_size, workers=4, run_config=run_config
    )
    print(f"  Test set size : {len(test_dataset)} RIRs")

    # ── Load speech files ─────────────────────────────────────────────────────
    print("\n── Loading LibriSpeech speech files ────────────────────────────────────")
    speech_files = load_speech_files(args.librispeech_path)

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\n── Running evaluation ──────────────────────────────────────────────────")
    print(
        f"  guidance_scale={args.guidance_scale}  |  steps={args.num_inference_steps}"
        f"  |  ddim={args.use_ddim}  |  batch={args.batch_size}"
    )

    all_real, all_gen, samples = run_evaluation(model, data_info, dereverb_model, test_dataloader, speech_files, device, args.guidance_scale, args.use_ddim, args.num_inference_steps, rng)

    if not all_real:
        print("\nNo valid samples were processed. Check data paths and signal lengths.")
        return

    # ── Report ────────────────────────────────────────────────────────────────
    real_agg = aggregate(all_real)
    gen_agg  = aggregate(all_gen)

    print_results_table(real_agg, gen_agg, n_samples=len(all_real))
    save_path.mkdir(parents=True, exist_ok=True)
    save_results(real_agg, gen_agg, all_real, all_gen, samples, args, save_path)


if __name__ == "__main__":
    main()
