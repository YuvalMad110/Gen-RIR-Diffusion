#!/usr/bin/env python3
"""
Precompute the acoustic measurements the dataset analysis needs, and store them as
columns in the mapping CSV:

    decay_len_s : seconds until the energy decay curve falls below db_cutoff, i.e. how
                  much of the RIR carries signal. Sets whether sample_max_sec is right.
    c50         : clarity, the ratio of energy in the first 50 ms to the rest, in dB.
    drr         : direct-to-reverberant ratio, in dB.

Reading the RIRs is the whole cost, so every metric that needs the waveform is computed
in that one pass. RT60 is not among them: 10_precompute_rt60.py already produced it, and
recomputing would put a column the training depends on at risk for analysis-only numbers.

Rows whose WAV is missing or empty are skipped, as are Matterport3D cross-region pairs,
which the dataset never serves. Rows where a metric cannot be estimated are left blank.

Run once per dataset. If the columns already exist the script exits without touching the
file (pass --force to recompute and overwrite).

Usage:
    python3 data/data_preparation/11_precompute_rir_metrics.py --scene-dataset matterport3d
    python3 data/data_preparation/11_precompute_rir_metrics.py --scene-dataset replica --force
"""

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.acoustic_metrics import compute_edc, compute_c50, compute_drr
from data.soundspaces import DATASET_DEFAULTS
from scene_stats import MAPPING_KEY, attach_columns

COLUMNS    = ['decay_len_s', 'c50', 'drr']
DB_CUTOFF  = -40.0    # the level decay_len_s measures down to
DRR_WINDOW = 2.5      # ms either side of the direct peak, matching compute_drr's default


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-dataset', choices=sorted(DATASET_DEFAULTS), required=True,
                        help='Which dataset to measure')
    parser.add_argument('--mapping-csv', default=None,
                        help="Path to the mapping CSV (default: the dataset's own)")
    parser.add_argument('--rir-root', default=None,
                        help="Root directory of SoundSpaces binaural RIRs (default: the dataset's own)")
    parser.add_argument('--force', action='store_true',
                        help='Recompute and overwrite existing columns')
    args = parser.parse_args()

    d = DATASET_DEFAULTS[args.scene_dataset]
    args.mapping_csv = args.mapping_csv or d['mapping_csv']
    args.rir_root    = args.rir_root    or d['rir_root']
    return args


def decay_length(rir: np.ndarray, sr: int) -> float:
    """Seconds until the energy decay curve first falls below DB_CUTOFF."""
    below = np.flatnonzero(compute_edc(rir) < DB_CUTOFF)   # compute_edc returns dB
    return float(below[0] / sr) if below.size else float(len(rir) / sr)


def main():
    args = parse_args()

    mapping = pd.read_csv(args.mapping_csv, comment='#')
    present = [c for c in COLUMNS if c in mapping.columns]
    if present and not args.force:
        print(f"{', '.join(present)} already present in {args.mapping_csv}. Nothing to do.")
        print('Pass --force to recompute and overwrite.')
        return

    rows = mapping[mapping['valid'] == True]
    if args.scene_dataset == 'matterport3d':
        rows = rows[rows['same_region'] == True]
    print(f'Measuring {len(rows):,} rows (skipping {len(mapping) - len(rows):,} the dataset never serves).')

    values = {c: {} for c in COLUMNS}
    failed, t0 = 0, time.time()
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc='Measuring RIRs'):
        waveform, sr = torchaudio.load(os.path.join(args.rir_root, row.rir_path))
        rir = waveform[int(row.best_channel)].numpy()
        if rir.size == 0:
            failed += 1
            continue
        key = (str(row.scene), str(row.receiver_idx), str(row.source_idx))
        for name, value in (('decay_len_s', decay_length(rir, sr)),
                            ('c50',         compute_c50(rir, sr)),
                            ('drr',         compute_drr(rir, sr))):
            if np.isfinite(value):
                values[name][key] = value

    elapsed = timedelta(seconds=int(time.time() - t0))
    print(f'\nMeasured in {elapsed}, {failed} rows held no audio.')

    attach_columns(args.mapping_csv, list(MAPPING_KEY), values, notes=[
        f'# decay_len_s: seconds until the energy decay curve falls below {DB_CUTOFF} dB',
        f'# c50: early-to-late energy ratio at 50 ms, dB. drr: direct-to-reverberant '
        f'ratio over +/-{DRR_WINDOW} ms around the direct peak, dB',
    ])


if __name__ == '__main__':
    main()
