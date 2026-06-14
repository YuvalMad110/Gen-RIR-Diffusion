"""
09_resample_soundspaces_22khz.py
---------------------------------
Resamples all SoundSpaces binaural RIRs from 44100 Hz to 22050 Hz and saves
them to a new root directory, preserving the original directory structure.

Skips files that already exist at the destination (safe to resume if interrupted).

Output: /dsi/fetaya-lab/yuvalmad/datasets/SoundSpaces/binaural_rirs/replica/22KHz/

Usage:
    python3 data/data_preparation/09_resample_soundspaces_22khz.py
    python3 data/data_preparation/09_resample_soundspaces_22khz.py --workers 8
"""

import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torchaudio
from tqdm import tqdm

SRC_ROOT = Path('/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica/')
DST_ROOT = Path('/dsi/fetaya-lab/yuvalmad/datasets/SoundSpaces/binaural_rirs/replica/22KHz/')
TARGET_SR = 22050


def _convert(args):
    src, dst = args
    if dst.exists():
        return 'skip'
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        waveform, sr = torchaudio.load(str(src))
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        torchaudio.save(str(dst), waveform, TARGET_SR)
        return 'ok'
    except Exception as e:
        return f'error: {src.name} — {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=min(16, cpu_count()),
                        help='Number of parallel worker processes (default: min(16, cpu_count))')
    parser.add_argument('--src', type=str, default=str(SRC_ROOT))
    parser.add_argument('--dst', type=str, default=str(DST_ROOT))
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    print(f"Source : {src_root}")
    print(f"Dest   : {dst_root}")
    print(f"Target : {TARGET_SR} Hz")
    print(f"Workers: {args.workers}")
    print("\nScanning source directory...", flush=True)

    all_wavs = sorted(src_root.rglob('*.wav'))
    total = len(all_wavs)
    print(f"Found {total:,} WAV files\n")

    pairs = [(p, dst_root / p.relative_to(src_root)) for p in all_wavs]

    errors = []
    skipped = 0
    converted = 0

    with Pool(args.workers) as pool:
        with tqdm(total=total, unit='file', dynamic_ncols=True) as pbar:
            for result in pool.imap_unordered(_convert, pairs, chunksize=64):
                if result == 'skip':
                    skipped += 1
                elif result == 'ok':
                    converted += 1
                else:
                    errors.append(result)
                pbar.update(1)
                pbar.set_postfix(converted=converted, skipped=skipped, errors=len(errors))

    print(f"\nDone.  converted={converted:,}  skipped={skipped:,}  errors={len(errors)}")
    if errors:
        print(f"\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == '__main__':
    main()
