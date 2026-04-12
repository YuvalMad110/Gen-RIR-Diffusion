"""
02_preprocess_soundspaces.py
-----------------------------
For each (scene, receiver, source) node pair in SoundSpaces/Replica, computes:
  - Horizontal bearing from receiver to source
  - best_angle: which of {0, 90, 180, 270} keeps the source most directly in front
  - best_channel: which binaural ear channel (0=left, 1=right) is closer to the source
  - Verifies the RIR file exists and records its relative path

Outputs a mapping CSV to data/soundspaces_replica_mapping.csv.
The training pipeline uses this file to load the correct RIR channel per sample,
and the rendering script uses it to know which image to pair with each RIR.

COORDINATE SYSTEM NOTE:
  points.txt uses a Z-up coordinate system (X and Y span the floor, Z is constant = height).
  SoundSpaces heading angles are defined in Habitat-Sim's Y-up space, where:
    heading=0   → receiver faces -Y direction (points.txt coords)
    heading=90  → receiver faces +X direction
    heading=180 → receiver faces +Y direction
    heading=270 → receiver faces -X direction
  The bearing calculation below is derived from this mapping.
  *** This assumption should be verified visually when rendering images. ***

Usage:
    python3 data/data_preparation/02_preprocess_soundspaces.py
    python3 data/data_preparation/02_preprocess_soundspaces.py \\
        --scenes office_1,room_0
"""

import argparse
import csv
import math
import pickle
from datetime import datetime
from pathlib import Path

ANGLES = [0, 90, 180, 270]

DEFAULT_METADATA_ROOT = "/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/metadata/replica"
DEFAULT_RIR_ROOT = "/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1]

CSV_COLUMNS = [
    "scene",
    "receiver_idx",   # target/listener position index (matches RIR filename first number)
    "source_idx",     # sound source position index (matches RIR filename second number)
    "bearing_deg",    # horizontal bearing from receiver to source [0, 360)
                      # 0° = source directly ahead at heading=0 (−Y direction in points.txt)
    "best_angle",     # selected from {0, 90, 180, 270} — closest to bearing
    "best_channel",   # 0=left ear, 1=right ear (ear physically closer to source)
    "receiver_x", "receiver_y", "receiver_z",
    "source_x", "source_y", "source_z",
    "distance_m",     # Euclidean 3D distance between receiver and source (meters)
    "rir_path",       # relative path: [scene]/[best_angle]/[receiver]_[source].wav
]


def parse_points(points_file: Path) -> dict:
    """Parse points.txt into {index: (x, y, z)}. Format: tab-separated index x y z."""
    positions = {}
    with open(points_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                idx = int(parts[0])
                positions[idx] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return positions


def compute_bearing(receiver_pos: tuple, source_pos: tuple) -> float | None:
    """
    Compute horizontal bearing from receiver to source in points.txt coordinates (Z-up).

    Returns angle in degrees [0, 360), measured clockwise from the heading=0 direction.
    In points.txt space: heading=0 corresponds to the -Y direction.
    So bearing=0   → source is in the -Y direction (directly ahead at angle=0)
       bearing=90  → source is in the +X direction (to the right at angle=0)
       bearing=180 → source is in the +Y direction (directly behind at angle=0)
       bearing=270 → source is in the -X direction (to the left at angle=0)

    Returns None if receiver and source are at the same horizontal position (self-pair).
    """
    dx = source_pos[0] - receiver_pos[0]
    dy = source_pos[1] - receiver_pos[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    # atan2(dx, -dy): 0° when source is in −Y direction, 90° when in +X direction
    bearing_rad = math.atan2(dx, -dy)
    return math.degrees(bearing_rad) % 360


def select_best_angle(bearing_deg: float) -> int:
    """
    Select the SoundSpaces angle folder {0, 90, 180, 270} closest to the bearing.
    Uses circular angular distance to handle wraparound (e.g. 350° vs 10°).
    """
    best_angle = 0
    min_diff = float("inf")
    for angle in ANGLES:
        diff = abs(((bearing_deg - angle) + 180) % 360 - 180)
        if diff < min_diff:
            min_diff = diff
            best_angle = angle
    return best_angle


def select_best_channel(bearing_deg: float, best_angle: int) -> int:
    """
    Select the binaural ear channel physically closer to the source.
    0 = left ear, 1 = right ear.

    delta = signed angular difference (bearing - best_angle) in [-180, 180]:
      delta > 0: source is to the right of the heading direction → right ear (1)
      delta < 0: source is to the left → left ear (0)
      delta == 0: source directly ahead → left ear (0, tiebreaker)
    """
    delta = ((bearing_deg - best_angle) + 180) % 360 - 180
    return 1 if delta > 0 else 0


def process_scene(
    scene_name: str,
    metadata_root: Path,
    rir_root: Path,
) -> tuple[list[dict], dict]:
    """
    Process all (receiver, source) pairs for a single scene.

    Returns:
        rows: list of CSV row dicts
        stats: summary statistics dict
    """
    meta_dir = metadata_root / scene_name
    rir_scene_dir = rir_root / scene_name

    # Load positions
    positions = parse_points(meta_dir / "points.txt")

    # Load graph — only nodes in the graph have RIR files
    with open(meta_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    graph_nodes = sorted(graph.nodes())

    rows = []
    n_missing = 0
    n_self_pairs = 0

    for receiver_idx in graph_nodes:
        for source_idx in graph_nodes:
            rx, ry, rz = positions[receiver_idx]
            sx, sy, sz = positions[source_idx]

            # Compute bearing
            bearing = compute_bearing((rx, ry, rz), (sx, sy, sz))
            if bearing is None:
                # Self-pair (same position) — skip
                n_self_pairs += 1
                continue

            best_angle = select_best_angle(bearing)
            best_channel = select_best_channel(bearing, best_angle)

            # Verify RIR file exists
            rir_rel = f"{scene_name}/{best_angle}/{receiver_idx}_{source_idx}.wav"
            rir_abs = rir_root / scene_name / str(best_angle) / f"{receiver_idx}_{source_idx}.wav"
            if not rir_abs.exists():
                n_missing += 1
                continue

            distance = math.dist((rx, ry, rz), (sx, sy, sz))

            rows.append({
                "scene": scene_name,
                "receiver_idx": receiver_idx,
                "source_idx": source_idx,
                "bearing_deg": round(bearing, 4),
                "best_angle": best_angle,
                "best_channel": best_channel,
                "receiver_x": rx, "receiver_y": ry, "receiver_z": rz,
                "source_x": sx, "source_y": sy, "source_z": sz,
                "distance_m": round(distance, 4),
                "rir_path": rir_rel,
            })

    stats = {
        "scene": scene_name,
        "n_graph_nodes": len(graph_nodes),
        "n_pairs_written": len(rows),
        "n_self_pairs_skipped": n_self_pairs,
        "n_missing_rirs": n_missing,
    }
    return rows, stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute mono-RIR mapping for SoundSpaces Replica dataset."
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
        help="Path to SoundSpaces metadata/replica directory",
    )
    parser.add_argument(
        "--rir-root",
        type=Path,
        default=DEFAULT_RIR_ROOT,
        help="Path to SoundSpaces binaural_rirs/replica directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the mapping CSV",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated scene names to process (default: all scenes in metadata root)",
    )
    args = parser.parse_args()

    if not args.metadata_root.exists():
        print(f"ERROR: metadata root not found: {args.metadata_root}")
        return
    if not args.rir_root.exists():
        print(f"ERROR: RIR root not found: {args.rir_root}")
        return

    # Discover scenes
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]
    else:
        scenes = sorted(d.name for d in args.metadata_root.iterdir() if d.is_dir())

    print(f"Processing {len(scenes)} scene(s)...\n")

    all_rows = []
    all_stats = []

    for scene in scenes:
        print(f"  {scene}...", end=" ", flush=True)
        rows, stats = process_scene(scene, args.metadata_root, args.rir_root)
        all_rows.extend(rows)
        all_stats.append(stats)
        print(
            f"{stats['n_pairs_written']} pairs"
            + (f" | {stats['n_missing_rirs']} missing RIRs" if stats["n_missing_rirs"] else "")
            + (f" | {stats['n_self_pairs_skipped']} self-pairs skipped" if stats["n_self_pairs_skipped"] else "")
        )

    # Write CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "soundspaces_replica_mapping.csv"

    with open(output_file, "w", newline="") as f:
        # File header comment
        f.write(f"# SoundSpaces Replica — Mono-RIR Mapping\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Metadata root: {args.metadata_root}\n")
        f.write(f"# RIR root: {args.rir_root}\n")
        f.write("#\n")
        f.write("# Column descriptions:\n")
        f.write("#   receiver_idx : target/listener position — matches first number in RIR filename\n")
        f.write("#   source_idx   : sound source position  — matches second number in RIR filename\n")
        f.write("#   bearing_deg  : horizontal bearing from receiver to source [0, 360)\n")
        f.write("#                  0°=source ahead at heading=0, 90°=source to right, etc.\n")
        f.write("#   best_angle   : SoundSpaces angle folder {0,90,180,270} closest to bearing\n")
        f.write("#                  This is the folder where the source is most directly in front\n")
        f.write("#   best_channel : 0=left ear, 1=right ear (ear physically closer to source)\n")
        f.write("#   distance_m   : Euclidean 3D distance receiver<->source in meters\n")
        f.write("#   rir_path     : relative path to the WAV file within the RIR root\n")
        f.write("#\n")
        f.write("# IMPORTANT: bearing_deg=0 aligns with SoundSpaces heading=0 (−Y in points.txt).\n")
        f.write("# Verify this assumption visually when rendering images with 03_render_replica_images.py.\n")
        f.write("#\n")

        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary
    total_pairs = sum(s["n_pairs_written"] for s in all_stats)
    total_missing = sum(s["n_missing_rirs"] for s in all_stats)
    print(f"\nTotal pairs written: {total_pairs:,}")
    if total_missing:
        print(f"WARNING: {total_missing} RIR files not found — check paths")
    print(f"Mapping saved to: {output_file}")

    # Print angle distribution
    angle_counts = {0: 0, 90: 0, 180: 0, 270: 0}
    channel_counts = {0: 0, 1: 0}
    for row in all_rows:
        angle_counts[row["best_angle"]] += 1
        channel_counts[row["best_channel"]] += 1
    print(f"\nAngle distribution (should be ~25% each if positions are spread evenly):")
    for a, c in angle_counts.items():
        print(f"  angle={a:>3}°: {c:>7,}  ({100*c/total_pairs:.1f}%)")
    print(f"Channel distribution:")
    print(f"  left  (0): {channel_counts[0]:>7,}  ({100*channel_counts[0]/total_pairs:.1f}%)")
    print(f"  right (1): {channel_counts[1]:>7,}  ({100*channel_counts[1]/total_pairs:.1f}%)")


if __name__ == "__main__":
    main()
