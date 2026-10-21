# Gen-RIR-Diffusion — Replica Experiments Log

> **Historical.** Every number here was measured with the earlier LSD, which took a single FFT of the whole signal truncated to 2048 samples and compared absolute levels — so it ignored everything past 128 ms and scored two RIRs with different tails as identical. We now consider that implementation wrong; the current frame-wise, gain-blind LSD is described in [`mp3d_model_experiments_log.md`](mp3d_model_experiments_log.md). The T60 windows and RIR alignment also differ, so these values are not comparable to the Matterport3D log and are kept for history only.

All models: 3-block UNet [32, 64, 128], STFT spectrogram (2×129×345), sr=22050 Hz, 100 epochs, guidance scale λ=2.0, 50-step DDIM.  
**Metrics:** T60 % = RT60 percentage error · DRR = DRR absolute error (dB) · LSD = Log Spectral Distance (dB). Lower is better for all.

---

## SoundSpaces + Replica Models (Active)

> 6 Replica scenes (office\_0–4, hotel\_0); binaural RIRs from SoundSpaces.  
> Scene-level split: 3 train / 1 val / 2 test → 9,722 / 2,256 / 1,982 pairs.  
> `scale_rir=True`, `apply_zero_tail=False`.

**Current golden config:** `model_config_VisualCond_HighRes_PosEnc512_MaskLoss.json`

| Run | Key differentiator | T60 % | DRR (dB) | LSD (dB) | C50 MAE (dB) |
|---|---|---|---|---|---|
| May17\_13-36-25\_dsief07 | First SoundSpaces run; images [56×74]; **no RT60** in condition vector | 15.71% | 5.55 | 12.14 | — |
| May28\_17-36-33\_dsief08 | **Scalar-only — no image conditioning** (ablation baseline); RT60 in cond | 15.85%† | 6.05 | 14.69 | — |
| May28\_17-51-17\_dsief08 | RT60 added to condition; images [56×74]; standard UNet arch | 18.55% | 5.94 | 12.62 | — |
| May28\_18-24-47\_dsief06 | HighRes UNet architecture (`VisualCond_HighRes.json`); still images [56×74] | 18.05% | 5.52 | 12.10 | — |
| Jun08\_19-28-08\_dsief06 | HighRes UNet + full-resolution images [392×518] + batch=32 | 15.48%† | 5.01 | 11.91 | — |
| Aug06\_19-26-54\_dsief08 | `src_trgt_dist_cond=True`; sample\_max\_sec=1 (129×345); batch=32; images [392×518] | 14.52%† | 4.61 | 12.69 | — |
| Aug06\_19-28-23\_dgx03 | Same as Aug06\_19-26-54 but batch=16 on dgx03 | — | — | — | 5.11 |
| Aug06\_19-41-44\_dsief08 | `src_trgt_dist_cond=True`; **sample\_max\_sec=0.6** (129×207); batch=8; images [392×518] | 8.83%† | 5.48 | 13.44 | 6.02 |
| Aug09\_18-18-41\_dsief07 | **`fusion_enable=False`** (direct concat, no FusionBlock); same as Aug06\_19-41-44 otherwise; batch=32; 50 epochs (best=50) | — | — | — | 4.79 |
| Aug10\_12-42-27\_dsief08 | **Ablation: no RT60** in condition (`use_rt60_condition=False`); `src_trgt_dist_cond=True`; sample\_max\_sec=0.6; batch=32 | 13.50%† | 4.71 | 11.59 | 3.48 |
| Aug10\_13-25-42\_dsief08 | **Ablation: no src\_trgt\_dist** (`src_trgt_dist_cond=False`); RT60 in condition; sample\_max\_sec=0.6; batch=32 | 12.19%† | 6.46 | 14.92 | — |
| Aug11\_23-51-50\_dsief06 | **First Fourier PE run**; `encoder_hidden_dims=[256,512]`; `cross_attention_dim=512`; PE output 143→256→512; batch=32 | 13.19%† | 4.37 | 13.02 | 5.08 |
| Aug13\_11-37-00\_dsief08 | **Ablation: no PE**, same `encoder_hidden_dims=[256,512]` as Aug11 — isolates PE contribution; batch=32; sample\_max\_sec=0.6 | 10.49%† | 5.67 | 12.43 | 5.93 |
| Aug13\_14-41-14\_dsief07 | Standard `[64,128]`, no PE; batch=8; sample\_max\_sec=0.6; 50 epochs (best=29) | 12.13%† | 5.48 | 12.58 | 5.06 |
| Aug17\_10-37-35\_dsief08 | Fourier PE; `encoder_hidden_dims=[256,512]`; `scale_rir=False`; no mask loss; batch=58; 70 epochs (best=52) | ⚠ 432%† | 6.31 | 13.00 | 9.91 |
| Aug17\_18-44-51\_dsief08 | Fourier PE; `encoder_hidden_dims=[256,512]`; `scale_rir=True`; **RT60-binary mask loss** (`db_threshold=−40`); batch=58; 70 epochs (best=24) | 13.97%† | 6.53 | 11.73 | 7.70 |

† T60 fit range (−5, −25 dB).

### Per-Octave-Band T60 % Error

*Re-run `full_model_eval.py` on a checkpoint to populate. Values are mean absolute % error.*

| Run | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---|---|---|---|---|---|

---

## External Baselines (SoundSpaces/Replica)

> Same test split as above (1,982 pairs). Evaluated via `Gen-RIR-Baselines/run_evaluation.py`.  
> T60 fit range (−5, −25 dB). Image2Reverb input: first room-overview RGB, resized to 224×224.

### Baseline Implementation Status

| Baseline | Adapter | Full eval (1982) | Notes |
|---|---|---|---|
| Image2Reverb | ✅ done | ⚠ only 31-sample debug on disk — need full run | `image2reverb/debug/eval_summary.txt` has 31 samples; numbers in table below may be from earlier full run |
| AV-RIR | ✅ done | ✅ done | `evaluations/replica_small_rooms/av_rir_zeros/` and `av_rir_gray/` — 1982 samples each |
| Rec-RIR | ✅ done | ❌ not run yet | adapter at `rec_rir/adapter.py`; mamba-ssm installed; run `python3 pre_generate.py --baseline rec_rir --debug-mode` |
| FAST-RIR | ❌ not written | ❌ | submodule at `fast_rir/FAST-RIR/`; **no data regeneration needed** — train directly on SoundSpaces dataloader (`image_root=None`, images off); input is `[mic_xyz, src_xyz, L, W, H, rt60]` normalized by /5−1 |
| MESH2IR | ❌ low priority | ❌ | requires mesh pointcloud input + trained on GWA; would need full retraining on SoundSpaces meshes |

### Metric Results

| Baseline | Input | T60 % | DRR Abs (dB) | LSD (dB) | C50 MAE (dB) |
|---|---|---|---|---|---|
| Image2Reverb | RGB image | 142.47% (median 117.12%) | 8.70 | 13.52 | 12.34 |
| AV-RIR zeros | Reverb speech + RGB image (geo-mat: zeros) | 38.86% (median 32.03%) | 4.76 | 10.27 | 5.42 |
| AV-RIR gray | Reverb speech + RGB image (geo-mat: gray 128) | 38.51% (median 31.87%) | 4.76 | 10.27 | 5.43 |

> **AV-RIR note:** Geo-mat images are unavailable for Replica (requires Tag2Text + Grounding DINO, not released). The two variants substitute a constant zero tensor and a constant gray (pixel=128) tensor respectively. Near-identical results confirm the geo-mat branch contributes only a fixed offset with a constant input. Speech conditioning: round-robin over LibriSpeech test-clean (`.flac`), resampled to 16 kHz, convolved with the reference RIR.

### Per-Octave-Band T60 % Error — Baselines

| Baseline | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---|---|---|---|---|---|
| Image2Reverb | 134.83% | 111.02% | 94.45% | 91.32% | 113.01% | 149.49% |
| AV-RIR zeros | 40.04% | 33.19% | 32.50% | 34.90% | 33.13% | 33.18% |
| AV-RIR gray | 39.23% | 32.21% | 31.33% | 33.90% | 32.74% | 32.84% |

---

## GTU\_RIR Models (Legacy)

> Trained on the internal GTU\_RIR dataset. No matched room images available; results are not comparable to SoundSpaces models.  
> **SoundSpaces training began with `May17_13-36-25_dsief07` (above).**

| Run | Key differentiator | T60 % | DRR (dB) | LSD (dB) |
|---|---|---|---|---|
| Aug06\_23-54-27\_dgx03 | First multi-GPU run; 4-block UNet; no CFG; `split_by_room=True` | — | — | — |
| Aug12\_12-09-09\_dgx03 | 4-block UNet; raw 10-dim cond via cross-attn (no encoder); no CFG | — | 2.66 | 12.91 |
| Aug12\_17-20-29\_dsief06 | Single-GPU replica of Aug12-dgx03 | — | — | — |
| Dec16\_17-33-21\_dsief07 | Reduced to 3-block UNet; still no encoder, no CFG | — | — | — |
| Dec18\_20-51-44\_dsief07 | First CFG-enabled model; added scalar MLP encoder (10→64→128) | — | 2.57 | 15.10 |
| Dec24\_20-02-59\_dsief07 | `apply_zero_tail=False` — best GTU model | 10.01% | 2.33 | 17.38 |
| Feb09\_12-05-44\_dsief08 | **Waveform-domain (1D UNet)** — experimental; catastrophic T60 error | 141.68% | 3.33 | 13.80 |

---

## New Entry Checklist

When adding a run, open `run_config.json` and `train.log` (first ~15 lines) and explicitly record every field below that differs from its default. Silence = default.

**`args`**
- `scenes` — default `null` (all 18)
- `use_rt60_condition` — default `true`
- `batch_size` · `epochs` (+ best epoch) · `lr` · `sample_max_sec` · `nSamples`
- `split_by_room` — default `false`
- `rir_view_type` · `room_overview_type` · `room_overview_config`
- `scale_rir` · `apply_zero_tail` · `db_cutoff` · `hop_length` · `n_fft` · `sr_target`

**`model_config`**
- `fusion_enable` — default `true`
- `src_trgt_dist_cond` — default `false`
- `loss_weighting` — default absent (uniform loss)
- `pos_encoder_cfg.enable` — default `false` (if present: note `n_freqs`, `encode_mode`, `include_raw`, `log_scale`)
- `encoder_hidden_dims` · `cross_attention_dim` · `input_cond_dim`
- `block_out_channels` · `layers_per_block` · `use_mid_block`
- `guidance_enabled` · `guidance_dropout_prob`
- `use_image_conditioning` · `image_fusion`

**`image_encoder_config`**
- `model_variant` · `feature_mode` · `last_n` · `layer_combination` · `image_size`
