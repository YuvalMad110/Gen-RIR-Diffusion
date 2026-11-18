# Architecture Changelog

Records every change that affects the model architecture.

**Scope column:**
- *Fixed* — change is always in effect; no flag, cannot be disabled
- *Configurable* — controlled by a flag in `model_config`; can be enabled/disabled per run

**Breaking column:**
- *Yes* — changes weight shapes or module presence; all checkpoints must record the
  controlling flag in `run_config.json → model_config`
- *No* — computation-only change; existing checkpoints load without modification

| Date | Scope | Breaking | Description |
|------|-------|----------|-------------|
| 2026-08-06 | Fixed | No | ~~Remove ReLU after last layer of condition encoder MLP~~ (reverted same day — ReLU restored; all trained models use ReLU on every layer) |
| 2026-08 | Configurable | Yes | Source-target distance appended to condition vector (`src_trgt_dist_cond`) |
| 2026-11 | Configurable | Yes | RIR representation and amplitude gain (`rir_domain`, `rir_gain`) |

---

## 2026-08

### [Fixed / Non-breaking] ~~Remove ReLU after last layer of condition encoder MLP~~ (reverted 2026-08-06)
Commit `547e205` removed ReLU from the last hidden layer of the condition encoder MLP.
Reverted manually the same day (before any Aug models launched) to keep the architecture
consistent with all prior trained models, ensuring fair comparison when evaluating
`src_trgt_dist_cond`. All trained models (May, Jun, Aug) use ReLU after every linear
layer including the last one. See Active TODOs in CLAUDE.md for planned revisit.

---

### [Configurable / Breaking] Source-target distance conditioning (`src_trgt_dist_cond`)
**Flag:** `src_trgt_dist_cond` (bool) in `model_config` section of `run_config.json`.

When `True`, the Euclidean distance `‖speaker_loc − mic_loc‖₂` is appended as an
explicit scalar to the conditioning vector, increasing `input_cond_dim` by 1
(e.g. 10 → 11 for SoundSpaces with RT60).

`build_condition_tensor` in `utils/inference_data_loading.py` is the single assembly
point for both training (`trainer.py`) and inference/evaluation scripts. It requires
`src_trgt_dist_cond` as a mandatory argument (no default) so misconfiguration fails
immediately at call time rather than silently using the wrong inputs.

**Checkpoints trained before this change** must have `"src_trgt_dist_cond": false`
manually added under `model_config` in their `run_config.json`.

Affected checkpoints patched:
- `Jun08_19-28-08_dsief06` — `src_trgt_dist_cond: false`
- `May28_17-36-33_dsief08` — `src_trgt_dist_cond: false`
- `May28_17-51-17_dsief08` — `src_trgt_dist_cond: false`
- `May28_18-24-47_dsief06` — `src_trgt_dist_cond: false`
- `May17_13-36-25_dsief07` — `src_trgt_dist_cond: false`

---

## 2026-11

### [Configurable / Breaking] RIR representation and amplitude gain (`rir_domain`, `rir_gain`)
**Flags:** `rir_domain` (`"spectrogram"` or `"waveform"`) and `rir_gain` (positive float) in the
`model_config` section of `run_config.json`.

`rir_domain` selects the signal the UNet diffuses on. `"spectrogram"` presents the RIR as its
two-channel real/imaginary STFT; `"waveform"` presents it as a `1 × T` image, which a 2D UNet
consumes unchanged because downsampling a height-1 axis is a no-op. It sets `in_channels`,
`out_channels` and `sample_size`, so it determines weight shapes.

`rir_gain` multiplies `x_0` before noising and divides the generated signal after denoising. The
schedule forms `x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps` with `eps` standard normal, so the share of
the schedule that carries any signal is set by the standard deviation of `x_0` relative to 1. A gain
of `1/median(sigma)` puts the median RIR at 1. `1.0` leaves the amplitude as the collate produced it.

`rir_to_model_input` and `model_output_to_rir` in `utils/signal_proc.py` are the single conversion
point for both directions, used by training and by every inference and evaluation script. Both
require `rir_gain` with no default, because a missing gain is silent: every metric this project
reports is scale-invariant — C50 and DRR are energy ratios, T60 and EDT are normalised-EDC slopes,
LSD is gain-blind, cosine similarity is scale-free — so an amplitude error would pass unmeasured.
Training alone may omit `rir_gain` from the config, where absence means no gain is applied, and the
resolved value is written into the saved `model_config` regardless.

**Checkpoints trained before this change** must have `"rir_domain": "spectrogram"` and
`"rir_gain": 1.0` added under `model_config` in their `run_config.json`. Both are the values those
runs used: the keys did not exist, so the signal was an STFT and no gain was applied.

Patched to `rir_domain: spectrogram`, `rir_gain: 1.0`: `Oct17_13-35-26_dsief08`,
`Oct17_13-43-44_dsief08`, `Oct22_21-36-55_dsief06`, `Oct24_01-07-43_dsief08`,
`Oct24_22-00-05_dsief08`, `Nov02_10-25-13_dsief08`, `Nov12_22-30-50_dsief08`,
`Nov12_22-33-30_dsief08`, `Nov16_01-16-23_dsief06`, `Nov16_01-26-46_dsief08`.

The per-run `configs/model_config*.json` snapshots are not patched; nothing reads them back, so they
stay as the record of what each run was given.
