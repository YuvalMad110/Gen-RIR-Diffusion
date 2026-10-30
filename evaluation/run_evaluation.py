"""
Score a set of pre-generated RIRs against the test reference store and write the results.

Generation already happened, so this runs on the CPU alone: a metric change costs a rescore
rather than another pass over the GPU. The generations are read from <run_dir>/generations/<tag>/
and the references from the store keyed by the same split_id, and the two are checked element by
element before a single number is computed — a plausible score from the wrong pairing is the one
failure that would not announce itself.

Results land in <run_dir>/evaluation/<tag>/, or <tag>_v2, _v3 when that directory already holds
an evaluation, so an earlier set of numbers is never overwritten.

--run-dir takes a bare run name, which is looked up under outputs/finished/, or a path to a run
directory anywhere.

    python3 evaluation/run_evaluation.py --run-dir Oct24_01-07-43_dsief08
    python3 evaluation/run_evaluation.py --run-dir Oct24_01-07-43_dsief08 --tag ddim200_g2_s42
    python3 evaluation/run_evaluation.py --run-dir outputs/finished/Oct24_01-07-43_dsief08 --limit 200
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from multiprocessing import Pool
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.build_test_references import load_test_references
from evaluation.lib.generation_utils import CONFIG_NAME as GENERATION_CONFIG
from evaluation.lib.generation_utils import generation_dir, load_generations
from evaluation.lib.metrics_utils import aggregate, score_pair
from evaluation.lib.result_plots import (format_title, plot_comparison, plot_edc_per_band,
                                         plot_histograms, select_samples)
from evaluation.lib.summary_report import write_summary
from utils.acoustic_metrics import DEFAULT_OCTAVE_BANDS, DEFAULT_T60_FIT_RANGE
from utils.misc import get_full_path

# =============================================================================
# Defaults
# =============================================================================
METRICS_NAME = 'metrics.csv'
SUMMARY_NAME = 'summary.txt'
CONFIG_NAME  = 'evaluation_config.json'

# Carried into the per-pair table ahead of the metrics, so a row can be traced to its room without joining anything back to the store.
IDENTITY_COLUMNS = ('sample_id', 'scene', 'room_id', 'room_type', 'distance_m')

DEFAULT_SPEECH = ('/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/test-clean/4992/41797/4992-41797-0013.flac')

# Comparison figures are drawn for each of these, the pairs ranked on it
DEFAULT_PLOT_METRICS = ('lsd_min', 't60_perc', 'c50_abs', 'drr_abs')

# Reverberant speech is rendered for each of these
DEFAULT_REVERB_METRICS = ('t60_perc', 'drr_abs')


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Score pre-generated RIRs against the test reference store')
    parser.add_argument('--run-dir', type=str, required=True, help='Run directory, or a bare folder name under outputs/finished/')
    parser.add_argument('--tag', type=str, default=None, help='Generation set to score; required only when the run holds more than one')
    parser.add_argument('--lsd-speech', type=str, default=DEFAULT_SPEECH, help='Clean speech convolved with both RIRs before the LSD')
    parser.add_argument('--octave-bands', type=float, nargs='+', default=DEFAULT_OCTAVE_BANDS, help='Centre frequencies for the per-band T60')
    parser.add_argument('--t60-fit-range', type=float, nargs=2, default=list(DEFAULT_T60_FIT_RANGE), help='dB window the T60 is fitted over')
    parser.add_argument('--limit', type=int, default=None, help='Score only the first N pairs, for a smoke test')
    parser.add_argument('--workers', type=int, default=8, help='Processes scoring pairs in parallel; 1 scores in this process')
    parser.add_argument('--plot-metrics', type=str, nargs='+', default=list(DEFAULT_PLOT_METRICS), help='Per-pair metrics to draw comparison and per-band figures for')
    parser.add_argument('--plot-samples', type=int, default=3, help='Pairs plotted per category, best / median / worst')
    parser.add_argument('--plot-max-sec', type=float, default=None, help='Clamp every plotted time axis to this many seconds')
    parser.add_argument('--reverb-metrics', type=str, nargs='+', default=list(DEFAULT_REVERB_METRICS), help='Metrics to render reverberant speech for, one pair per category')
    parser.add_argument('--speech4reverb', type=str, nargs='+', default=[DEFAULT_SPEECH], help='Clean speech files the reverberant samples are rendered from')
    parser.add_argument('--no-plots', action='store_true', help='Score and write the tables without drawing figures')
    parser.add_argument('--no-reverb-speech', action='store_true', help='Skip rendering the reverberant speech samples')

    args = parser.parse_args()
    args.run_dir = get_full_path(args.run_dir, 'outputs/finished')
    return args


# =============================================================================
# Preconditions
# =============================================================================

def _set_summary(set_dir: Path) -> str:
    """What one generation set holds, for the chooser's listing."""
    try:
        with open(set_dir / GENERATION_CONFIG) as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return 'unreadable config'

    created = str(config.get('created', '?')).replace('T', ' ')
    return f"{config.get('n_samples', 0):,} samples  ·  generated {created}"


def choose_tag(root: Path, names: list) -> str:
    """
    Ask which generation set to score, listing what each one holds.

    Only a terminal is asked. Anywhere else — a detached or piped run — this refuses and prints
    the same listing, so the job fails where it can be read instead of blocking on a prompt
    nobody is there to answer.
    """
    listing = [f'  {i:>2}. {name:<26} {_set_summary(root / name)}' for i, name in enumerate(names, 1)]
    if not sys.stdin.isatty():
        raise SystemExit(f'--tag is required: {root} holds {len(names)} generation sets\n'
                         + '\n'.join(listing))

    print(f'\n{root} holds {len(names)} generation sets:')
    print('\n'.join(listing))

    while True:
        try:
            answer = input(f'  which one? [1-{len(names)}, q to quit] ').strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit('\nnothing scored')

        if answer.lower() in ('q', 'quit'):
            raise SystemExit('nothing scored')
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print(f'  not a choice between 1 and {len(names)}')


def resolve_tag(run_dir, tag):
    """
    Which generation set to score, taken from --tag, from the one set that is there, or asked.

    Naming the sets matters more than the refusal itself: the tags are derived from sampler
    settings, so the list is also the answer to which settings have been generated.
    """
    if tag:
        return tag

    root = Path(run_dir) / 'generations'
    sets = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not sets:
        raise SystemExit(f'no generation sets under {root}; run pre_generate.py first')
    if len(sets) == 1:
        print(f'  --tag not given; scoring the only set present, {sets[0]}')
        return sets[0]
    return choose_tag(root, sets)


def check_generations_match_store(generations: dict, references: dict) -> None:
    """
    Confirm the two sets describe the same pairs, in the same order, before anything is scored.

    A mismatch names the first index that differs and both ids there, because the failure this
    guards against produces numbers rather than an error.
    """
    gen_ids = [str(s) for s in generations['sample_id']]
    ref_ids = [str(s) for s in references['sample_id']]
    if len(gen_ids) > len(ref_ids):
        raise ValueError(f'the generation set holds {len(gen_ids):,} samples but the store holds '
                         f'{len(ref_ids):,}; it cannot be a subset of this split')

    for i, (generated, reference) in enumerate(zip(gen_ids, ref_ids)):
        if generated != reference:
            raise ValueError(f'sample_ids differ at index {i:,}: generations have {generated!r}, '
                             f'store has {reference!r}. The two were built from different splits.')

    scope = 'the whole split' if len(gen_ids) == len(ref_ids) else f'the first {len(gen_ids):,} of {len(ref_ids):,}'
    print(f'  alignment ok: {len(gen_ids):,} sample_ids match element-wise, {scope}')


def load_dry_signal(speech_path: str, sr: int):
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


def score_split(generations: dict, references: dict, sr: int, dry_signal, octave_bands,
                t60_fit_range, limit=None, workers: int = 1) -> list:
    """
    Score every pair and return the per-pair rows, in the order the pairs are stored.

    Each pair is scored independently, so the work splits across processes with no effect on the
    result: the pool preserves order and every measurement reads only its own pair. Each row
    carries its identity columns ahead of the metrics, so the table stands alone.

    Returns:
        A list of flat dicts, one per pair.
    """
    count = len(generations['sample_id']) if not limit else min(limit, len(generations['sample_id']))
    identities = [{column: references[column][i] for column in IDENTITY_COLUMNS} for i in range(count)]
    tasks = [(generations['generated'][i], references['references'][i],
              {key: references[key][i] for key in ('rt60', 'c50', 'drr')}) for i in range(count)]

    started = time.time()
    if workers > 1:
        with Pool(workers, initializer=_init_worker,
                  initargs=(sr, dry_signal, t60_fit_range, octave_bands)) as pool:
            scored = list(tqdm(pool.imap(_score_one, tasks, chunksize=8),
                               desc='Scoring', total=count))
    else:
        _init_worker(sr, dry_signal, t60_fit_range, octave_bands)
        scored = [_score_one(task) for task in tqdm(tasks, desc='Scoring', total=count)]

    elapsed = time.time() - started
    print(f'  scored {count:,} pairs in {elapsed / 60:.1f} min ({elapsed / max(count, 1):.2f}s each, '
          f'{workers} worker{"s" if workers > 1 else ""})')
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
# Writing
# =============================================================================

def resolve_output_dir(run_dir, tag: str) -> Path:
    """
    Where this evaluation's results go, versioned so an earlier run's numbers survive.

    The version is one past the highest already there, not the first free number, so a deleted
    evaluation leaves a gap rather than having its number handed to a later run.

    Returns:
        <run_dir>/evaluation/<tag>, or <tag>_v<highest + 1>.
    """
    base = Path(run_dir) / 'evaluation' / tag
    if not base.exists():
        return base

    versions = [int(suffix) for path in base.parent.glob(f'{tag}_v*')
                for suffix in [path.name.rsplit('_v', 1)[-1]] if suffix.isdigit()]
    return base.parent / f'{tag}_v{max(versions, default=1) + 1}'


def write_metrics_csv(path: Path, rows: list) -> Path:
    """Write the per-pair table through a temp path and a rename."""
    frame = pd.DataFrame(rows)
    ordered = list(IDENTITY_COLUMNS) + [c for c in frame.columns if c not in IDENTITY_COLUMNS]

    staging = path.with_suffix('.csv.tmp')
    frame[ordered].to_csv(staging, index=False)
    os.replace(staging, path)
    return path


def build_header_fields(generation_config: dict, tag: str, speech_path: str) -> dict:
    """The summary's header, in the order it is printed, with 'Model' first."""
    c = generation_config
    return {'Model': Path(c['run_dir']).name,
            'Generations': tag,
            'Sampler': f"{c['sampler']}, {c['num_inference_steps']} steps, "
                       f"guidance {c['guidance_scale']}, seed {c['seed']}",
            'Un-scaling': c['unscaling'],
            'Split': c['split_id'],
            'LSD signal': f'speech convolved — {Path(speech_path).name}'}


def write_results(target_dir: Path, rows: list, stats: dict, generation_config: dict, tag: str,
                  args, n_total: int) -> None:
    """Write the per-pair table, the summary and the configuration that produced them."""
    target_dir.mkdir(parents=True, exist_ok=True)

    write_metrics_csv(target_dir / METRICS_NAME, rows)
    write_summary(str(target_dir / SUMMARY_NAME), stats,
                  build_header_fields(generation_config, tag, args.lsd_speech),
                  len(rows), n_total, args.octave_bands)

    config = {'tag': tag, 'split_id': generation_config['split_id'],
              'run_dir': generation_config['run_dir'],
              'lsd_speech': args.lsd_speech, 'speech4reverb': list(args.speech4reverb),
              'octave_bands': list(args.octave_bands),
              't60_fit_range': list(args.t60_fit_range), 'n_scored': len(rows),
              'n_split': n_total, 'created': datetime.now().isoformat(timespec='seconds'),
              'generation': generation_config}

    staging = target_dir / f'{CONFIG_NAME}.tmp'
    with open(staging, 'w') as handle:
        json.dump(config, handle, indent=2)
    os.replace(staging, target_dir / CONFIG_NAME)


# =============================================================================
# Figures
# =============================================================================

def _title_base(generation_config: dict, tag: str) -> dict:
    """The parts every figure title carries, before whatever that figure adds."""
    return {'Model': Path(generation_config['run_dir']).name, '': tag,
            'steps': generation_config.get('num_inference_steps'),
            'guidance': generation_config.get('guidance_scale')}


def write_figures(target_dir: Path, rows: list, generations: dict, references: dict, sr: int,
                  generation_config: dict, tag: str, args) -> None:
    """
    A comparison and a per-band figure for each plotted metric, and one histogram grid.

    Each metric ranks its own pairs, so the comparison for C50 shows the pairs C50 is worst on
    rather than the pairs some other metric chose.
    """
    figure_dir = target_dir / 'figures'
    edc_dir = figure_dir / 'edc'
    edc_dir.mkdir(parents=True, exist_ok=True)
    base = _title_base(generation_config, tag)

    drawn = []
    for metric in args.plot_metrics:
        selected = select_samples(rows, metric, args.plot_samples)
        if not selected:
            print(f'  no finite {metric} to rank on; skipped its figures')
            continue

        plot_comparison(selected, generations, references, rows, sr,
                        format_title({**base, 'selected by': metric}),
                        figure_dir / f'{metric}_comparison.png',
                        max_sec=args.plot_max_sec, fit_range=tuple(args.t60_fit_range))
        plot_edc_per_band(selected, generations, references, rows, sr,
                          format_title({**base, 'EDC per band, selected by': metric}),
                          edc_dir / f'{metric}_edc_per_band.png',
                          octave_bands=args.octave_bands, max_sec=args.plot_max_sec,
                          fit_range=tuple(args.t60_fit_range))
        drawn.append(metric)

    plot_histograms(rows, format_title({**base, 'pairs': f'{len(rows):,}'}),
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


def write_reverb_speech(target_dir: Path, rows: list, generations: dict, references: dict,
                        sr: int, metrics, speech_paths) -> int:
    """
    Render the best, median and worst pair of each metric as reverberant speech, to listen to.

    The clean speech sits beside the metric folders, so a sample can always be compared with
    what went into it. One pair per category: these are listening examples, not a ranking.

    Returns:
        How many wav files were written, the clean copies included.
    """
    reverb_dir = target_dir / 'reverb_speech'
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


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ---------- Resolve what is being scored ----------
    tag = resolve_tag(args.run_dir, args.tag)
    source_dir = generation_dir(args.run_dir, tag)
    generations = load_generations(source_dir)
    generation_config = generations['config']
    sr = generation_config['sr_target']

    print(f'\nRun {Path(args.run_dir).name}  ·  split {generation_config["split_id"]}  ·  tag {tag}')
    print(f'  generations: {len(generations["sample_id"]):,} of {generations["generated"].shape[1]:,} samples')
    print(f'  bands {[int(b) for b in args.octave_bands]}  ·  T60 fitted over {tuple(args.t60_fit_range)} dB')

    # ---------- Check the two sides describe the same pairs ----------
    references = load_test_references(generation_config['split_id'])
    check_generations_match_store(generations, references)
    dry_signal = load_dry_signal(args.lsd_speech, sr)

    # ---------- Score ----------
    print(f'\nScoring {len(generations["sample_id"]):,} pairs'
          + (f', stopping at {args.limit:,}' if args.limit else ''))
    rows = score_split(generations, references, sr, dry_signal, args.octave_bands,
                       tuple(args.t60_fit_range), limit=args.limit, workers=args.workers)

    stats = aggregate([{k: v for k, v in row.items() if k not in IDENTITY_COLUMNS} for row in rows])
    report_failures(stats, len(rows))

    # ---------- Write ----------
    target_dir = resolve_output_dir(args.run_dir, tag)
    write_results(target_dir, rows, stats, generation_config, tag, args, len(references['sample_id']))
    print(f'\nWrote {target_dir}')
    print(f'  {METRICS_NAME}  {SUMMARY_NAME}  {CONFIG_NAME}')

    # ---------- Draw and render ----------
    if not args.no_plots:
        write_figures(target_dir, rows, generations, references, sr, generation_config, tag, args)
    if not args.no_reverb_speech:
        write_reverb_speech(target_dir, rows, generations, references, sr,
                            args.reverb_metrics, args.speech4reverb)


if __name__ == '__main__':
    main()
