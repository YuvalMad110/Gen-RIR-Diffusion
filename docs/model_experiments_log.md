# Gen-RIR-Diffusion — Experiments Log

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

† T60 fit range (−5, −25 dB).

### Per-Octave-Band T60 % Error

*Re-run `full_model_eval.py` on a checkpoint to populate. Values are mean absolute % error.*

| Run | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---|---|---|---|---|---|

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
