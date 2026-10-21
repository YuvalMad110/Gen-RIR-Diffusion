# Gen-RIR-Diffusion — Matterport3D Experiments Log

Every run below trains on the same Matterport3D split and is scored by the same metrics. The second column of the model table names the single thing that differs from the golden configuration; everything else is shared and listed under [Golden configuration](#golden-configuration).

Lower is better for every metric. T60 % shows the mean with the median in brackets, since its distribution is skewed.

## Results

### Our models

| Run | Changed vs. golden | T60 % | DRR (dB) | LSD (dB) | C50 MAE (dB) |
|---|---|---|---|---|---|
| Oct11_17-21-16_dsief08 | — (golden) | 8.77 (6.93) | 4.90 | 6.61 | 7.18 |

### Baselines

> Same test split, 42,633 pairs. One row per baseline: the variant chosen as the fairest comparison. The golden model repeats here for reading convenience.
> **Dataset** names the corpora a baseline needs beyond Matterport3D itself.

| Model | Conditioning | Dataset | T60 % | DRR (dB) | LSD (dB) | C50 MAE (dB) |
|---|---|---|---|---|---|---|
| **Ours (golden)** | RGB images + room geometry + RT60 | — | **8.77 (6.93)** | 4.90 | **6.61** | 7.18 |
| Image2Reverb | Single RGB image | — | 706.12 (558.46) | 8.81 | 9.24 | 13.45 |
| AV-RIR | Reverberant speech + RGB image | LibriSpeech test-clean | 60.46 (42.27) | **4.83** | 7.46 | **4.83** |
| Rec-RIR | Reverberant speech | VCTK + LibriSpeech (speech), NOISEX-92 (noise) | *training in progress* | | | |

### Per-octave-band T60 % error

| Model | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---|---|---|---|---|---|
| Ours (golden) | 19.35 | 17.44 | 15.65 | 10.73 | 11.77 | 12.69 |
| Image2Reverb | 831.59 | 675.27 | 565.61 | 514.73 | 500.18 | 540.33 |
| AV-RIR | 60.82 | 50.36 | 51.62 | 58.65 | 62.78 | 66.08 |

## Golden configuration

`Oct11_17-21-16_dsief08` · git `e2675bf` · config `model_config_VisualCond_MP3D.json`

**Data** — Matterport3D, 189,393 / 35,469 / 42,633 pairs across 1,117 rooms, split by room with seed 42 and balanced on room count. Per-room cap 1,000. Filters: `min_rt60` 0.1 s, `max_scaled_peak` 65,504, void 0.4 indoor / 0.8 outdoor / 0.5 per room. 16 kHz, `sample_max_sec` 0.6 s, `n_fft` 256, `hop_length` 64. RIRs scaled by `exp(k·t)` from the stored RT60, no zero-tail.

**Conditioning** — room dimensions, microphone and speaker positions, RT60, and source-receiver distance; five RGB images per pair (one receiver→source view, four room overviews) at 224×294 through a DA3 ViT-L pyramid encoder, `last_n` 4, fused.

**Model** — 3-block UNet [32, 64, 128] over a 2×129×151 spectrogram, 1,000 train timesteps, classifier-free guidance with dropout 0.1, Fourier positional encoding on the scalar conditions, uniform loss.

**Training** — 30 epochs, batch 72, lr 3e-4, best at epoch 28.

**Inference** — 50-step DDIM, guidance 2.0, half precision.

## Metrics

T60 is fitted over −15 to −40 dB of the energy decay curve and extrapolated to −60 dB. DRR uses a ±1.25 ms window around the direct peak, counting everything after it as reverberant. LSD is measured on clean speech convolved with each RIR, with the generated RIR energy-matched to the reference first, so it reads spectral shape and not level; it is frame-wise over three STFT resolutions. Generated and reference RIRs are padded to the longer of the two, never truncated.

Dry speech for LSD: `LibriSpeech/test-clean/4992/41797/4992-41797-0013.flac`, the same utterance for every model.

Changing any of this means re-running every evaluation in this file and rewriting this section — old numbers are not kept.

## Reading the numbers

- **T60 partly measures envelope compliance.** The generated RIR is un-scaled using the RT60 it was conditioned on, so some of the agreement is imposed by that inverse transform rather than predicted. It is a fair measure of whether the model honours its conditioning, not of blind RT60 estimation.
- **Cosine similarity near zero is expected** and is why it is not in the table. RIR fine structure is phase-like and not reproducible sample by sample; a generative model should not be penalised for that.
- **LSD is gain-blind by construction**, so a model that gets the spectral shape right but the overall level wrong scores well here. Level error shows up in DRR and C50 instead.
- **C50 and DRR are not independent.** Both compare early to late energy, so a model that misplaces the direct-to-reverberant balance moves both together.
- **Image2Reverb's published T60 is not comparable to this table.** Its test flow measures both RIRs after a magnitude-only STFT round trip that discards the phase; on Matterport3D that round trip alone shifts RT60 by 21.5%. See `Gen-RIR-Baselines/BASELINE_FIDELITY.md`.

## New entry checklist

Open `run_config.json` and record every field that differs from the golden configuration above. Silence means identical.

**`args`** — `epochs` (+ best epoch) · `batch_size` · `lr` · `sample_max_sec` · `sr_target` · `n_fft` · `hop_length` · `scale_rir` · `apply_zero_tail` · `use_rt60_condition` · `rir_view_type` · `room_overview_type` · the five quality filters · `max_pairs_per_room` · `balance_split_by` · `split_by_room` · `random_seed`

**`model_config`** — `pos_encoder_cfg.enable` (and `n_freqs`, `encode_mode`, `include_raw`, `log_scale` when on) · `src_trgt_dist_cond` · `fusion_enable` · `loss_weighting` · `encoder_hidden_dims` · `block_out_channels` · `layers_per_block` · `guidance_enabled` · `guidance_dropout_prob`

**`image_encoder_config`** — `model_variant` · `feature_mode` · `last_n` · `layer_combination` · `image_size`

Also record the git hash, and confirm the evaluation used the metrics described above. A run whose `inputs` fingerprint no longer matches the mapping on disk cannot be compared here and should not be added.
