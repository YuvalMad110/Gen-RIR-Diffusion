# Evaluation Guide

All evaluation scripts live in `evaluation/`. Run them from the project root:

```bash
python3 evaluation/full_model_eval.py --model_dir outputs/finished/<run_name>
```

---

## Scripts

| Script | Dataset | Purpose |
|---|---|---|
| `full_model_eval.py` | GTU + SoundSpaces | Core acoustic metrics (T60, DRR, EDT, C50, LSD, EDC) on the test split; histograms and representative sample plots |
| `guidance_sweep.py` | GTU + SoundSpaces | Sweep CFG guidance scale, rank scales by metric; visualises top scales |
| `dereverb_eval.py` | GTU + SoundSpaces | Dereverberation probe (PESQ / STOI / SI-SDR) — checks whether generated RIRs behave like real ones when fed to a dereverberation model |
| `synthetic_eval.py` | GTU only | Generates RIRs from randomly sampled conditions within GTU room-geometry ranges; used as an unconditional quality probe |
| `full_model_eval_legacy.py` | GTU only | Legacy version of `full_model_eval.py` for old checkpoints that lack `run_config.json` |
| `guidance_sweep_legacy.py` | GTU only | Legacy version of `guidance_sweep.py` for old checkpoints |

---

## Quick Start

### Modern runs (have `run_config.json`)

```bash
# Full evaluation on test set
CUDA_VISIBLE_DEVICES=0 python3 evaluation/full_model_eval.py \
    --model_dir outputs/finished/May17_13-36-25_dsief07

# Guidance sweep
CUDA_VISIBLE_DEVICES=0 python3 evaluation/guidance_sweep.py \
    --model_dir outputs/finished/May17_13-36-25_dsief07 \
    --guidance_min 1.0 --guidance_max 6.0 --guidance_step 0.5

# Dereverberation probe
CUDA_VISIBLE_DEVICES=0 python3 evaluation/dereverb_eval.py \
    --diffusion_model_path May17_13-36-25_dsief07 \
    --librispeech_path /path/to/LibriSpeech/Test
```

### Legacy GTU runs (no `run_config.json`)

```bash
python3 evaluation/full_model_eval_legacy.py \
    --model_path outputs/finished/Dec24_20-02-59_dsief07/model_best.pth.tar \
    --dataset_path ./datasets/GTU_rir/GTU_RIR.pickle.dat
```

---

## Script Reference

### `full_model_eval.py`

**Inputs:**
- `--model_dir` — run output directory
- `--nSamples` — override test set size (default: all)
- `--guidance_scale` — CFG scale (default: 2.0)
- `--num_inference_steps` — denoising steps (default: 50)
- `--use_ddim` — use DDIM scheduler
- `--batch_size` — default: 16
- `--baseline_method` — `habets` / `pra` / `none` (default: habets)
- `--speech_path` — clean speech WAV for reverbed LSD

**Outputs** (inside `<model_dir>/evaluation_<timestamp>/`):
- `evaluation_summary.txt` — mean ± std per metric, comparison table
- `detailed_metrics.csv` — per-sample metrics
- `histograms_summary.png`, `histograms/` — metric distribution plots
- `selected_samples_*.pt` + visualisation PNGs — representative RIR pairs

---

### `guidance_sweep.py`

**Inputs:**
- `--model_dir` — run output directory
- `--guidance_min/max/step` — scale range (default: 1.0–8.0, step 0.5)
- `--sweep_metrics` — metrics to rank (default: t60_perc c50 drr lsd)
- `--vis_metric` — metric for sample selection (default: t60_perc)
- `--n_top_scales` — number of top scales to plot (default: 3)

**Outputs** (inside `<model_dir>/guidance_sweep_<timestamp>/`):
- `sweep_summary.txt` — ranking tables for both methods
- `sweep_rankings.csv`, `sweep_all_metrics.csv`
- `sweep_samples.png` — waveform + EDC grid for representative samples

---

### `dereverb_eval.py`

**Inputs:**
- `--diffusion_model_path` — run folder name, full folder path, or full path to `model_best.pth.tar`
- `--librispeech_path` — LibriSpeech test directory
- `--dereverb_model` — `nara_wpe` (default, no deps) or `unet`
- `--guidance_scale`, `--num_inference_steps`, `--use_ddim`

**Outputs** (inside `<model_dir>/dereverb_eval_<timestamp>/`):
- `dereverb_eval_results.txt` — PESQ / STOI / SI-SDR comparison table
- `dereverb_eval_boxplots.png`
- `samples/` — audio files (clean, reverb, dereverb)

---

### `synthetic_eval.py` (GTU only)

Samples random room conditions from GTU geometry ranges, generates RIRs, and computes acoustic properties. Run from the project root:

```bash
python3 evaluation/synthetic_eval.py --model_path outputs/finished/<run>/model_best.pth.tar
```

---

## Shared Utilities

| Helper | Location | Purpose |
|---|---|---|
| `load_pretrained_model(run_dir, device)` | `utils/inference_data_loading.py` | Load model + run_config from a modern run directory |
| `data_params_from_run_config(run_config)` | `utils/inference_data_loading.py` | Extract data_info-compatible dict from run_config['args'] |
| `build_test_dataloader(data_params, batch_size, workers, run_config)` | `utils/inference_data_loading.py` | Reconstruct test-split DataLoader (GTU or SoundSpaces) |
| `build_condition_tensor(batch, device)` | `utils/inference_data_loading.py` | Build scalar conditioning tensor from a batch dict or tuple |
| `evaluate_rir_pair` | `utils/acoustic_metrics.py` | Per-pair T60/DRR/EDT/C50/LSD/EDC computation |
| `aggregate_metrics` | `utils/acoustic_metrics.py` | Aggregate list of per-pair metrics into mean ± std |

---

## Legacy Support

Old GTU checkpoints (trained before `run_config.json` was added) use the `_legacy` scripts:

- `full_model_eval_legacy.py` — pass `--model_path` and `--dataset_path`
- `guidance_sweep_legacy.py` — pass `--model_path` and `--dataset_path`

These scripts read `data_info` directly from the checkpoint. They do not support image-conditioned models.
