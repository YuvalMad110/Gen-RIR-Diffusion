"""
05d_void_log.py
----------------
Records the void fraction of every rendered Matterport3D image.

A render shows black wherever the camera ray hits no mesh — open sky above an
outdoor region, gaps the Matterport scanner never covered, or the empty space
around a degenerate room. Measuring that share gives a per-image quality signal
the dataset class can filter on, without changing what gets rendered.

Covers both render sets:

  overviews  {image_root}/room_overviews/{scene}_r{region:03d}/corner{1-4}_rgb.jpg
             → data/matterport3d/room_overviews_void_log.csv
             columns: scene, region_idx, corner, void_frac

  pairs      {image_root}/target2source_rgb/{scene}/tgt{receiver}_src{source}.jpg
             → data/matterport3d/target2source_void_log.csv
             columns: scene, receiver_idx, source_idx, void_frac

Safe to re-run at any time; it derives everything from the images on disk, so the
log stays complete regardless of how many render passes produced them.

Usage:
    python3 data/data_preparation/05d_void_log.py                    # overviews
    python3 data/data_preparation/05d_void_log.py --set pairs        # ~966K images, 20-40 min
    python3 data/data_preparation/05d_void_log.py --set both
    python3 data/data_preparation/05d_void_log.py --scenes 17DRP5sb8fy,1LXtFkjw3qL
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from scene_stats import attach_columns

DEFAULT_IMAGE_ROOT = Path("/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/rendered_images")
OUTPUT_DIR         = Path(__file__).resolve().parents[1] / "matterport3d"

# A pixel counts as void when its brightest channel falls below this value. Rendered
# void is exactly 0; the margin absorbs JPEG ringing around high-contrast edges.
VOID_LEVEL = 12

_ROOM_DIR_RE = re.compile(r"^(?P<scene>.+)_r(?P<region>\d+)$")
_CORNER_RE   = re.compile(r"^corner(?P<corner>\d+)_rgb\.jpg$")
_PAIR_RE     = re.compile(r"^tgt(?P<receiver>\d+)_src(?P<source>\d+)\.jpg$")


def void_fraction(path: Path) -> float:
    """Share of pixels in one image with no mesh behind them."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return float((rgb.max(axis=2) < VOID_LEVEL).mean())


def _progress(done: int, total: int, t0: float, every: int) -> None:
    """Report against the total image count, so the ETA covers the whole run."""
    if done % every and done != total:
        return
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done else 0
    print(f"  {done:,}/{total:,} images ({100 * done / total:5.1f}%)  |  "
          f"{timedelta(seconds=int(elapsed))} elapsed  |  ETA {timedelta(seconds=int(eta))}",
          flush=True)


def collect_overviews(image_root: Path, scenes: set | None) -> list:
    """Return sorted (scene, region_idx, corner, void_frac) rows."""
    root = image_root / "room_overviews"
    if not root.is_dir():
        raise SystemExit(f"No room_overviews directory under {image_root}")

    dirs = []
    for d in root.iterdir():
        m = _ROOM_DIR_RE.match(d.name) if d.is_dir() else None
        if m and (not scenes or m.group("scene") in scenes):
            dirs.append((m.group("scene"), int(m.group("region")), d))
    dirs.sort(key=lambda x: (x[0], x[1]))

    print(f"  indexing {len(dirs):,} room directories...", end=" ", flush=True)
    work = []
    for scene, region, d in dirs:
        for img in sorted(d.iterdir()):
            m = _CORNER_RE.match(img.name)
            if m:
                work.append((scene, region, int(m.group("corner")), img))
    print(f"{len(work):,} images", flush=True)

    rows, t0 = [], time.time()
    every = max(1, len(work) // 50)
    for scene, region, corner, img in work:
        rows.append((scene, region, corner, void_fraction(img)))
        _progress(len(rows), len(work), t0, every)
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


def collect_pairs(image_root: Path, scenes: set | None) -> list:
    """Return sorted (scene, receiver_idx, source_idx, void_frac) rows."""
    root = image_root / "target2source_rgb"
    if not root.is_dir():
        raise SystemExit(f"No target2source_rgb directory under {image_root}")

    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and (not scenes or d.name in scenes))

    print(f"  indexing {len(dirs):,} scene directories...", end=" ", flush=True)
    work = []
    for d in dirs:
        for img in sorted(d.iterdir()):
            m = _PAIR_RE.match(img.name)
            if m:
                work.append((d.name, int(m.group("receiver")), int(m.group("source")), img))
    print(f"{len(work):,} images", flush=True)

    rows, t0 = [], time.time()
    every = max(1, len(work) // 50)
    for scene, receiver, source, img in work:
        rows.append((scene, receiver, source, void_fraction(img)))
        _progress(len(rows), len(work), t0, every)
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


SETS = {
    "overviews": {
        "collect":  collect_overviews,
        "csv":      "room_overviews_void_log.csv",
        "title":    "MP3D room-overview void log",
        "columns":  "scene,region_idx,corner,void_frac",
        "join":     "Join to room_geometry.csv on (scene, region_idx) for room label and dimensions.",
        "advice":   ("# Rooms are judged on the mean void_frac of their four corners, written to\n"
                     "# room_geometry.csv as mean_overview_void. Aggregating per room rather than\n"
                     "# dropping individual corners keeps the fixed four-view structure the model\n"
                     "# relies on."),
    },
    "pairs": {
        "collect":  collect_pairs,
        "csv":      "target2source_void_log.csv",
        "title":    "MP3D target2source pair-view void log",
        "columns":  "scene,receiver_idx,source_idx,void_frac",
        "join":     "Join to soundspaces_matterport3d_mapping.csv on (scene, receiver_idx, source_idx).",
        "advice":   ("# Each row is one training sample's conditioning image, so a threshold here drops\n"
                     "# individual pairs rather than whole rooms."),
    },
}


def write_log(rows: list, spec: dict, out_csv: Path, image_root: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# {spec['title']}",
        f"# Generated:  {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"# Image root: {image_root}",
        "#",
        "# void_frac is the fraction of pixels in a render that have no mesh behind them,",
        "# computed as the share of pixels whose brightest RGB channel is below",
        f"# {VOID_LEVEL}. It rises with open sky above outdoor regions, gaps the Matterport",
        "# scanner never covered, and views into empty space, so it serves as a per-image",
        "# quality signal. Range 0.0 (fully covered) to 1.0.",
        "#",
        "# One row per rendered image, derived solely from the images on disk.",
        f"# {spec['join']}",
        "#",
        spec["advice"],
        spec["columns"],
    ]
    lines = [f"{a},{b},{c},{v:.4f}" for a, b, c, v in rows]
    out_csv.write_text("\n".join(header + lines) + "\n")


def read_log(out_csv: Path) -> list:
    """Rows of an existing log, so the columns can be attached without rescanning images."""
    with open(out_csv) as f:
        data = [l for l in f if not l.startswith("#")][1:]
    return [(p[0], int(p[1]), int(p[2]), float(p[3]))
            for p in (l.rstrip("\n").split(",") for l in data)]


def summarize(rows: list, label: str) -> None:
    v = np.array([r[3] for r in rows])
    print(f"\n{label}: {len(v):,} images")
    print("void_frac distribution:")
    for p in (50, 75, 90, 95, 99):
        print(f"   p{p:<3d} {np.percentile(v, p):.3f}")
    print(f"   max  {v.max():.3f}")
    print()
    for thr in (0.20, 0.30, 0.40, 0.50):
        print(f"   void_frac > {thr:.2f} : {int((v > thr).sum()):7,} images "
              f"({100 * (v > thr).mean():4.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log the void fraction of every rendered MP3D image."
    )
    parser.add_argument("--set", choices=["overviews", "pairs", "both"], default="overviews",
                        help="Which render set to scan (default: overviews)")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT,
                        help="Root containing room_overviews/ and target2source_rgb/")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory the CSVs are written to")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene names (default: every scene found)")
    parser.add_argument("--from-logs", action="store_true",
                        help="Take void_frac from the existing logs instead of reading the "
                             "images, and only attach the columns the dataset reads")
    args = parser.parse_args()

    scenes = {s.strip() for s in args.scenes.split(",")} if args.scenes else None
    targets = ["overviews", "pairs"] if args.set == "both" else [args.set]

    for name in targets:
        spec = SETS[name]
        out_csv = args.output_dir / spec["csv"]

        if args.from_logs:
            print(f"\nReading {name} void from {out_csv}")
            rows = read_log(out_csv)
        else:
            print(f"\nScanning {name} under {args.image_root}")
            rows = spec["collect"](args.image_root, scenes)
            if not rows:
                print(f"  no {name} images found, skipping")
                continue
            write_log(rows, spec, out_csv, args.image_root)
        summarize(rows, name)

        # Hand the measurement to the two files the dataset actually reads
        note = f"# void_frac: share of pixels whose brightest channel is below {VOID_LEVEL}"
        if name == "pairs":
            values = {(s, str(r), str(c)): v for s, r, c, v in rows}
            attach_columns(args.output_dir / "soundspaces_matterport3d_mapping.csv",
                           ["scene", "receiver_idx", "source_idx"], {"void_frac": values},
                           notes=[note])
        else:
            per_room = defaultdict(list)
            for scene, region, _corner, v in rows:
                per_room[(scene, str(region))].append(v)
            values = {k: sum(v) / len(v) for k, v in per_room.items()}
            attach_columns(args.output_dir / "room_geometry.csv",
                           ["scene", "region_idx"], {"mean_overview_void": values},
                           notes=[note + ", averaged over a room's four overviews"])


if __name__ == "__main__":
    main()
