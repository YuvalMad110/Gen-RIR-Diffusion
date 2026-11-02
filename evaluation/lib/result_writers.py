"""
Scoring a prepared set of pairs and writing everything an evaluation leaves behind.

The diffusion flow and the baselines differ in where their pairs come from — a reference store
keyed by split_id on one side, a saved archive on the other — and in what identifies a pair. They
do not differ in anything after that, so the pool, the per-pair table, the figures and the
reverberant speech live here and are called with one set of rules by every flow.

Two arguments carry the difference. The caller builds the `(generated, reference,
reference_metrics)` tasks, which is what lets one flow read its reference values from a store
while another measures them from the signal. And `identity_columns` names the columns that go
ahead of the metrics, which differ with what a flow knows about a pair.

Everything is written through a temp path and a rename, so an interrupted run never leaves a
half-written table where a complete one is expected.
"""

import os
import time
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from evaluation.lib.metrics_utils import score_pair
from evaluation.lib.result_plots import (format_title, plot_comparison, plot_edc_per_band,
                                         plot_histograms, select_samples)


# =============================================================================
# Inputs
# =============================================================================

def load_dry_signal(speech_path: str, sr: int) -> np.ndarray:
    """The clean speech both signals are convolved with before an LSD, resampled to sr."""
    signal, _ = librosa.load(speech_path, sr=sr)
    print(f'  speech: {Path(speech_path).name}, {len(signal) / sr:.2f}s at {sr} Hz')
    return np.asarray(signal, dtype=np.float64)


# =============================================================================
# Scoring
# =============================================================================

# Settings a worker reuses for every pair, filled once per process by the pool's initializer
_WORKER = {}


def _init_worker(sr: int, dry_signal, t60_fit_range, octave_bands) -> None:
    """Give this process the settings every pair is scored with."""
    _WORKER.update(sr=sr, dry_signal=dry_signal, t60_fit_range=t60_fit_range,
                   octave_bands=octave_bands)


def _score_one(task: tuple) -> dict:
    """One pair, against the settings this process was initialised with."""
    generated, reference, reference_metrics = task
    return score_pair(generated, reference, reference_metrics, _WORKER['sr'],
                      dry_signal=_WORKER['dry_signal'], t60_fit_range=_WORKER['t60_fit_range'],
                      octave_bands=_WORKER['octave_bands'])


def score_rows(tasks: list, identities: list, sr: int, dry_signal, octave_bands, t60_fit_range,
               workers: int = 1) -> list:
    """
    Score every prepared pair and return the per-pair rows, in the order they were given.

    Each pair is scored independently, so the work splits across processes with no effect on the
    result: the pool preserves order and every measurement reads only its own pair. Each row
    carries its identity columns ahead of the metrics, so the table stands alone.

    Args:
        tasks:         (generated, reference, reference_metrics) per pair, the metrics dict
                       holding 'rt60', 'c50' and 'drr' however the caller obtained them
        identities:    what names each pair, one dict per task and in the same order
        sr:            sample rate
        dry_signal:    speech convolved with both signals before an LSD
        octave_bands:  centre frequencies for the per-band T60
        t60_fit_range: dB window the T60 is fitted over
        workers:       processes scoring in parallel; 1 scores in this process

    Returns:
        A list of flat dicts, one per pair, each identity merged ahead of its metrics.
    """
    if len(tasks) != len(identities):
        raise ValueError(f'{len(tasks):,} tasks against {len(identities):,} identities; '
                         f'every pair must carry exactly one identity')

    started = time.time()
    if workers > 1:
        with Pool(workers, initializer=_init_worker,
                  initargs=(sr, dry_signal, t60_fit_range, octave_bands)) as pool:
            scored = list(tqdm(pool.imap(_score_one, tasks, chunksize=8),
                               desc='Scoring', total=len(tasks)))
    else:
        _init_worker(sr, dry_signal, t60_fit_range, octave_bands)
        scored = [_score_one(task) for task in tqdm(tasks, desc='Scoring', total=len(tasks))]

    elapsed = time.time() - started
    print(f'  scored {len(tasks):,} pairs in {elapsed / 60:.1f} min '
          f'({elapsed / max(len(tasks), 1):.2f}s each, {workers} worker'
          f'{"s" if workers > 1 else ""})')
    return [{**identity, **metrics} for identity, metrics in zip(identities, scored)]


def report_failures(stats: dict, n_scored: int) -> None:
    """Name every metric that could not be measured on part of the split, or say that none could not."""
    failed = {key: n_scored - entry['n'] for key, entry in stats.items()
              if isinstance(entry, dict) and entry.get('n', n_scored) < n_scored}
    if not failed:
        print(f'  every metric measured on all {n_scored:,} pairs')
        return

    listed = ', '.join(f'{key} {count:,}' for key, count in sorted(failed.items(), key=lambda kv: -kv[1]))
    print(f'  metrics with failures ({len(failed)}): {listed}')


# =============================================================================
# Tables
# =============================================================================

def resolve_output_dir(parent, name: str) -> Path:
    """
    Where this evaluation's results go, versioned so an earlier run's numbers survive.

    The version is one past the highest already there, not the first free number, so a deleted
    evaluation leaves a gap rather than having its number handed to a later run.

    Returns:
        <parent>/<name>, or <name>_v<highest + 1> when that is taken.
    """
    parent = Path(parent)
    base = parent / name
    if not base.exists():
        return base

    versions = [int(suffix) for path in parent.glob(f'{name}_v*')
                for suffix in [path.name.rsplit('_v', 1)[-1]] if suffix.isdigit()]
    return parent / f'{name}_v{max(versions, default=1) + 1}'


def write_metrics_csv(path, rows: list, identity_columns) -> Path:
    """Write the per-pair table, its identity columns first, through a temp path and a rename."""
    path = Path(path)
    frame = pd.DataFrame(rows)
    ordered = list(identity_columns) + [c for c in frame.columns if c not in identity_columns]

    staging = path.with_suffix('.csv.tmp')
    frame[ordered].to_csv(staging, index=False)
    os.replace(staging, path)
    return path


# =============================================================================
# Figures
# =============================================================================

def write_figures(target_dir, rows: list, generations: dict, references: dict, sr: int,
                  title_base: dict, plot_metrics, plot_samples: int, plot_max_sec,
                  octave_bands, t60_fit_range) -> None:
    """
    A comparison and a per-band figure for each plotted metric, and one histogram grid.

    Each metric ranks its own pairs, so the comparison for C50 shows the pairs C50 is worst on
    rather than the pairs some other metric chose.

    Args:
        target_dir:    the evaluation directory; figures land in figures/ beneath it
        rows:          the per-pair rows the selection ranks on
        generations:   {'generated': indexable by row position}
        references:    {'references': indexable by row position, 'sample_id': the same}
        sr:            sample rate
        title_base:    the parts every title carries, before whatever each figure adds
        plot_metrics:  metrics to rank and draw, each getting its own pair of figures
        plot_samples:  pairs drawn per category, best / median / worst
        plot_max_sec:  clamp every time axis to this many seconds, or None for the full signal
        octave_bands:  centre frequencies the per-band EDC figure covers
        t60_fit_range: dB window drawn as the fit guides
    """
    figure_dir = Path(target_dir) / 'figures'
    edc_dir = figure_dir / 'edc'
    edc_dir.mkdir(parents=True, exist_ok=True)

    drawn = []
    for metric in plot_metrics:
        selected = select_samples(rows, metric, plot_samples)
        if not selected:
            print(f'  no finite {metric} to rank on; skipped its figures')
            continue

        plot_comparison(selected, generations, references, rows, sr,
                        format_title({**title_base, 'selected by': metric}),
                        figure_dir / f'{metric}_comparison.png',
                        max_sec=plot_max_sec, fit_range=tuple(t60_fit_range))
        plot_edc_per_band(selected, generations, references, rows, sr,
                          format_title({**title_base, 'EDC per band, selected by': metric}),
                          edc_dir / f'{metric}_edc_per_band.png',
                          octave_bands=octave_bands, max_sec=plot_max_sec,
                          fit_range=tuple(t60_fit_range))
        drawn.append(metric)

    plot_histograms(rows, format_title({**title_base, 'pairs': f'{len(rows):,}'}),
                    figure_dir / 'metric_histograms.png')
    print(f'  figures for {", ".join(drawn) if drawn else "no metric"}, plus the histogram grid')


# =============================================================================
# Reverberant speech
# =============================================================================

def _write_pair(directory: Path, category: str, speech: str, reference_wet, generated_wet,
                sr: int) -> None:
    """
    One pair of reverberant signals, scaled together.

    Both are divided by whichever of the two peaks is larger, so neither clips and the level
    the model produced stays audible against the reference instead of being normalised away.
    """
    peak = max(np.abs(reference_wet).max(), np.abs(generated_wet).max())
    scale = 0.95 / peak if peak > 0 else 1.0
    sf.write(directory / f'{category}_ref_{speech}.wav', reference_wet * scale, sr, subtype='PCM_16')
    sf.write(directory / f'{category}_gen_{speech}.wav', generated_wet * scale, sr, subtype='PCM_16')


def write_reverb_speech(target_dir, rows: list, generations: dict, references: dict,
                        sr: int, metrics, speech_paths) -> int:
    """
    Render the best, median and worst pair of each metric as reverberant speech, to listen to.

    The clean speech sits beside the metric folders, so a sample can always be compared with
    what went into it. One pair per category: these are listening examples, not a ranking.

    Returns:
        How many wav files were written, the clean copies included.
    """
    reverb_dir = Path(target_dir) / 'reverb_speech'
    reverb_dir.mkdir(parents=True, exist_ok=True)

    speech = {}
    for path in speech_paths:
        signal, _ = librosa.load(path, sr=sr)
        stem = Path(path).stem
        speech[stem] = np.asarray(signal, dtype=np.float64)
        sf.write(reverb_dir / f'{stem}.wav', speech[stem], sr, subtype='PCM_16')

    written = len(speech)
    for metric in metrics:
        selected = select_samples(rows, metric, per_category=1)
        if not selected:
            print(f'  no finite {metric} to rank on; rendered no speech for it')
            continue

        metric_dir = reverb_dir / metric
        metric_dir.mkdir(exist_ok=True)
        for category, indices in selected.items():
            index = indices[0]
            reference = np.asarray(references['references'][index], dtype=np.float64).flatten()
            generated = np.asarray(generations['generated'][index], dtype=np.float64).flatten()
            print(f'    {metric} {category}: {references["sample_id"][index]}')

            for stem, dry in speech.items():
                _write_pair(metric_dir, category, stem,
                            np.convolve(dry, reference), np.convolve(dry, generated), sr)
                written += 2

    print(f'  reverb speech: {written} wav files under {reverb_dir.name}/')
    return written
