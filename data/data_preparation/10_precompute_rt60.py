#!/usr/bin/env python3
"""
Precompute RT60 for every SoundSpaces pair the dataset can draw and store it as an
rt60 column in that dataset's mapping CSV.

Rows whose WAV file is empty or missing (valid=False) are skipped, as are Matterport3D
cross-region pairs, which the dataset never serves. Valid rows where T60 estimation
fails (EDC doesn't decay far enough) are kept but their rt60 cell is left as NaN — they
remain usable for regular training but are excluded when --use-rt60-condition is active.

Per-scene RT60 NaN counts are written to the dataset's scene-stats file so the scale of
the problem is visible at a glance.

Run once per dataset before --use-rt60-condition training. If the 'rt60' column already
exists, the script exits without touching the file (pass --force to recompute and
overwrite).

Usage:
    python3 data/data_preparation/10_precompute_rt60.py --scene-dataset replica
    python3 data/data_preparation/10_precompute_rt60.py --scene-dataset matterport3d
    python3 data/data_preparation/10_precompute_rt60.py --scene-dataset replica --force
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.acoustic_metrics import compute_t60
from data.soundspaces import DATASET_DEFAULTS, _room_keys, _subsample_rooms
from scene_stats import merge_notes, write_scene_stats

_STATS_DIR = {'replica': 'replica', 'matterport3d': 'matterport3d'}


def fit_range(text):
    """
    Parse a dB range given as two comma-separated values, such as '-15,-40'. Either order
    is accepted. Returns (upper, lower), the order compute_t60 takes.

    Comma-separated rather than two arguments because argparse reads a leading minus as
    the start of another flag; for the same reason the option must be written with an
    equals sign, --fit-range=-15,-40.
    """
    try:
        first, second = (float(part) for part in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected two dB values as 'hi,lo', got {text!r}")
    if first == second:
        raise argparse.ArgumentTypeError(f'range needs two different values, got {text!r}')
    return max(first, second), min(first, second)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-dataset', choices=sorted(DATASET_DEFAULTS), required=True,
                        help='Which dataset to compute RT60 for')
    parser.add_argument('--mapping-csv', default=None,
                        help="Path to the mapping CSV (default: the dataset's own)")
    parser.add_argument('--rir-root', default=None,
                        help="Root directory of SoundSpaces binaural RIRs (default: the dataset's own)")
    parser.add_argument('--capped-only', action='store_true',
                        help='Cover only the rows the per-room cap keeps, which trains on the same set for a fraction of the RIRs; leaves the rest NaN')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Seed the per-room cap draws from; must match the training seed')
    parser.add_argument('--fit-range', type=fit_range, default=None,
                        help="dB range the decay line is fitted over, as --fit-range=-15,-40 (default: the dataset's t60_fit_range). The equals sign is required, since a leading minus otherwise reads as another flag")
    parser.add_argument('--force', action='store_true',
                        help='Recompute and overwrite existing rt60 column')
    args = parser.parse_args()

    d = DATASET_DEFAULTS[args.scene_dataset]
    args.mapping_csv  = args.mapping_csv or d['mapping_csv']
    args.rir_root     = args.rir_root    or d['rir_root']
    args.geometry_csv = d['geometry_csv']
    # The config is the declared window for this dataset; it travels into run_config.json
    # from there, so overriding it here without updating it leaves the two disagreeing.
    args.fit_range    = args.fit_range or tuple(d['t60_fit_range'])
    args.stats_file  = (Path(project_root) / 'data' / _STATS_DIR[args.scene_dataset]
                        / 'analysis' / 'soundspaces_scene_stats.txt')
    return args


def region_counts(geometry_csv: str) -> dict:
    """Regions per scene, which only the geometry CSV knows: a region holding no
    navigable node never reaches the mapping."""
    if not os.path.exists(geometry_csv):
        return {}
    g = pd.read_csv(geometry_csv, comment='#')
    return g.groupby('scene').size().to_dict()


def main():
    args = parse_args()

    mapping = pd.read_csv(args.mapping_csv, comment='#')

    if 'rt60' in mapping.columns and not args.force:
        print(f"'rt60' column already exists in {args.mapping_csv}. Nothing to do.")
        print("Pass --force to recompute and overwrite.")
        return

    valid_mask = mapping['valid'] == True
    if args.scene_dataset == 'matterport3d':
        valid_mask &= mapping['same_region'] == True
    valid_rows = mapping[valid_mask]
    n_skipped  = (~valid_mask).sum()
    print(f"Processing {len(valid_rows):,} rows (skipping {n_skipped:,} the dataset never serves).")

    cap = DATASET_DEFAULTS[args.scene_dataset]['max_pairs_per_room']
    if args.capped_only and cap is not None:
        # The cap is idempotent, so covering just the rows it keeps gives training the
        # same set it would get from a full pass. _orig carries the mapping index through.
        valid_rows = valid_rows.assign(_orig=valid_rows.index)
        valid_rows = _subsample_rooms(valid_rows, _room_keys(valid_rows, args.scene_dataset),
                                      cap, args.random_seed).set_index('_orig')
        print(f"  --capped-only: narrowed to {len(valid_rows):,} rows at cap={cap}, seed={args.random_seed}.")

    rt60_series = pd.Series(np.nan, index=mapping.index, dtype=float)
    n_nan = 0

    for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Computing RT60"):
        rir_path = os.path.join(args.rir_root, row['rir_path'])
        waveform, sr = torchaudio.load(rir_path)   # [2, T] stereo

        channel = int(row['best_channel'])
        rir = waveform[channel].numpy()             # [T]

        t60 = compute_t60(rir, sr, fit_range=args.fit_range)
        if np.isnan(t60):
            print(f"  [T60 NaN] {row['scene']} / {row['rir_path']} (EDC too short/dry — skipped)")
            n_nan += 1
            continue

        rt60_series[idx] = t60

    mapping['rt60'] = rt60_series

    # Write mapping CSV, keeping the comment header and recording how rt60 was produced.
    # The fit range is the one setting that cannot be recovered from the column itself,
    # and it is the one most likely to change, so it belongs beside the values.
    hi, lo = args.fit_range
    note = (f'# rt60: reverberation time in seconds. Least-squares fit over the {hi:g}..{lo:g} dB\n'
            f'#       range of the energy decay curve, extrapolated to -60 dB. Measured on the\n'
            f'#       whole recording at its native sample rate, with no sample_max_sec crop.')

    with open(args.mapping_csv, 'r') as f:
        comment_lines = [l for l in f if l.startswith('#')]

    with open(args.mapping_csv, 'w') as f:
        f.writelines(merge_notes(comment_lines, [note]))
        f.write(mapping.to_csv(index=False))

    n_ok = len(valid_rows) - n_nan
    print(f"\nDone. RT60 computed for {n_ok:,} rows, {n_nan} NaN (T60 estimation failed).")
    print(f"Mapping saved to {args.mapping_csv}.")

    if args.capped_only and cap is not None:
        print(f"\nThis was a --capped-only pass, so the rows beyond the cap are still NaN and the"
              f"\nscene stats are left alone — their NaN counts would read as estimation failures."
              f"\nCover the rest with: --scene-dataset {args.scene_dataset} --force")
    else:
        write_scene_stats(mapping, args.scene_dataset, args.stats_file,
                          region_counts(args.geometry_csv))


if __name__ == '__main__':
    main()
