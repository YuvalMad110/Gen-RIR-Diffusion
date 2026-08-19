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
