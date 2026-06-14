#!/usr/bin/env python3
"""
Precompute RT60 for all SoundSpaces/Replica pairs and store as a column in
soundspaces_replica_mapping.csv.

Rows whose WAV file is empty or missing (valid=False) are skipped. Valid rows
where T60 estimation fails (EDC doesn't decay far enough) are kept but their
rt60 cell is left as NaN — they remain usable for regular training but are
excluded when --use-rt60-condition is active.

Per-scene RT60 NaN counts are appended to soundspaces_scene_stats.txt so the
scale of the problem is visible at a glance.

Run once before --use-rt60-condition training. If the 'rt60' column already
exists, the script exits without touching the file (pass --force to recompute
and overwrite).

Usage:
    python3 data/data_preparation/10_precompute_rt60.py
    python3 data/data_preparation/10_precompute_rt60.py --force
    python3 data/data_preparation/10_precompute_rt60.py \
        --mapping-csv data/soundspaces_replica_mapping.csv \
        --rir-root /dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica
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

_DEFAULT_MAPPING_CSV = os.path.join(project_root, 'data', 'soundspaces_replica_mapping.csv')
_DEFAULT_RIR_ROOT    = '/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica'
_STATS_FILE          = Path(project_root) / 'data' / 'data_analysis' / 'soundspaces_scene_stats.txt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping-csv', default=_DEFAULT_MAPPING_CSV,
                        help='Path to soundspaces_replica_mapping.csv')
    parser.add_argument('--rir-root', default=_DEFAULT_RIR_ROOT,
                        help='Root directory of SoundSpaces binaural RIRs')
    parser.add_argument('--force', action='store_true',
                        help='Recompute and overwrite existing rt60 column')
    return parser.parse_args()


def update_scene_stats(mapping: pd.DataFrame, stats_path: Path) -> None:
    """Regenerate soundspaces_scene_stats.txt from the mapping DataFrame."""
    scenes = sorted(mapping['scene'].unique())
    rows = []
    for scene in scenes:
        sm = mapping[mapping['scene'] == scene]
        n_valid   = int((sm['valid'] == True).sum())
        n_invalid = int((sm['valid'] != True).sum())
        n_total   = len(sm)
        valid_rows = sm[sm['valid'] == True]
        n_rt60_nan = int(valid_rows['rt60'].isna().sum()) if 'rt60' in sm.columns else None
        rows.append((scene, n_valid, n_invalid, n_rt60_nan, n_total))

    total_valid   = sum(r[1] for r in rows)
    total_invalid = sum(r[2] for r in rows)
    total_nan     = sum(r[3] for r in rows if r[3] is not None)
    total_total   = sum(r[4] for r in rows)

    lines = [
        "SoundSpaces/Replica — RIR pairs per scene",
        "==========================================",
        f"{'Scene':<28} {'Valid':>8} {'Invalid':>8} {'RT60 NaN':>10} {'Total':>8}",
        "-" * 66,
    ]
    for scene, n_valid, n_invalid, n_nan, n_total in rows:
        nan_str = f"{n_nan:>10,}" if n_nan is not None else f"{'—':>10}"
        lines.append(f"{scene:<28} {n_valid:>8,} {n_invalid:>8,} {nan_str} {n_total:>8,}")
    lines += [
        "-" * 66,
        f"{'TOTAL':<28} {total_valid:>8,} {total_invalid:>8,} {total_nan:>10,} {total_total:>8,}",
        "",
        "GTU dataset: 15,202 samples (single pickle, in-memory)",
        "",
        "Notes:",
        "  - Each pair = one (receiver, source) directed RIR",
        "  - Invalid  = WAV file missing or empty (0 bytes)",
        "  - RT60 NaN = valid WAV where T60 estimation failed (EDC too short/dry)",
        "              these rows are usable for regular training but excluded",
        "              from --use-rt60-condition runs",
        "  - SoundSpaces samples all positions at a fixed ear height (~1.5m above floor)",
        "  - All 18 scenes are single-floor in the acoustic data",
    ]
    stats_path.write_text("\n".join(lines) + "\n")
    print(f"Scene stats updated: {stats_path}")


def main():
    args = parse_args()

    mapping = pd.read_csv(args.mapping_csv, comment='#')

    if 'rt60' in mapping.columns and not args.force:
        print(f"'rt60' column already exists in {args.mapping_csv}. Nothing to do.")
        print("Pass --force to recompute and overwrite.")
        return

    valid_mask = mapping['valid'] == True
    valid_rows = mapping[valid_mask]
    n_invalid  = (~valid_mask).sum()
    print(f"Processing {len(valid_rows):,} valid rows (skipping {n_invalid:,} invalid).")

    rt60_series = pd.Series(np.nan, index=mapping.index, dtype=float)
    n_nan = 0

    for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Computing RT60"):
        rir_path = os.path.join(args.rir_root, row['rir_path'])
        waveform, sr = torchaudio.load(rir_path)   # [2, T] stereo

        channel = int(row['best_channel'])
        rir = waveform[channel].numpy()             # [T]

        t60 = compute_t60(rir, sr)
        if np.isnan(t60):
            print(f"  [T60 NaN] {row['scene']} / {row['rir_path']} (EDC too short/dry — skipped)")
            n_nan += 1
            continue

        rt60_series[idx] = t60

    mapping['rt60'] = rt60_series

    # Write mapping CSV (preserve comment header)
    with open(args.mapping_csv, 'r') as f:
        comment_lines = [l for l in f if l.startswith('#')]

    with open(args.mapping_csv, 'w') as f:
        f.writelines(comment_lines)
        f.write(mapping.to_csv(index=False))

    n_ok = len(valid_rows) - n_nan
    print(f"\nDone. RT60 computed for {n_ok:,} rows, {n_nan} NaN (T60 estimation failed).")
    print(f"Mapping saved to {args.mapping_csv}.")

    update_scene_stats(mapping, _STATS_FILE)


if __name__ == '__main__':
    main()
