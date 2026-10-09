"""
02_preprocess_mp3d.py
----------------------
For each (scene, receiver, source) node pair in SoundSpaces/Matterport3D, computes:
  - Horizontal bearing from receiver to source
  - best_angle: which of {0, 90, 180, 270} keeps the source most directly in front
  - best_channel: which binaural ear channel (0=left, 1=right) is closer to the source
  - Verifies the RIR file exists and records its relative path
  - same_region: whether receiver and source fall within the same .house region polygon

Outputs a mapping CSV to data/matterport3d/soundspaces_matterport3d_mapping.csv.
The training pipeline uses this file to load the correct RIR channel per sample.

COORDINATE SYSTEM NOTE:
  points.txt uses the same Z-up coordinate system as Replica. Navigable points
  sit at z ≈ 1.52 m (ear height above floor), vs. Replica's z ≈ 0.009 m.
  The bearing calculation is identical to 02_preprocess_soundspaces.py.

Usage:
    python3 data/data_preparation/02_preprocess_mp3d.py
    python3 data/data_preparation/02_preprocess_mp3d.py --scenes 17DRP5sb8fy,1LXtFkjw3qL
"""

import argparse
import csv
import math
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from scene_stats import MIN_RIR_BYTES, carry_over_columns, write_scene_stats

ANGLES = [0, 90, 180, 270]

DEFAULT_METADATA_ROOT = "/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/metadata/mp3d"
DEFAULT_RIR_ROOT      = "/dsi/fetaya-lab/yuvalmad/datasets/SoundSpaces/binaural_rirs/mp3d"
DEFAULT_MESH_ROOT     = "/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/meshes"
DEFAULT_OUTPUT_DIR    = Path(__file__).resolve().parents[1] / "matterport3d"

CSV_COLUMNS = [
    "scene",
    "receiver_idx",    # target/listener position index (matches RIR filename first number)
    "source_idx",      # sound source position index (matches RIR filename second number)
    "bearing_deg",     # horizontal bearing from receiver to source [0, 360)
    "best_angle",      # selected from {0, 90, 180, 270} — closest to bearing
    "best_channel",    # 0=left ear, 1=right ear (ear physically closer to source)
    "receiver_x", "receiver_y", "receiver_z",
    "source_x", "source_y", "source_z",
    "distance_m",      # Euclidean 3D distance between receiver and source (meters)
    "rir_path",        # relative path: [scene]/data/binaural_rirs/mp3d/[scene]/[angle]/[r]_[s].wav
    "valid",           # True if the WAV exists and holds at least MIN_RIR_BYTES; False otherwise
    "same_region",     # True if receiver and source fall within the same .house floor polygon
    "receiver_region", # integer region index from .house (-1 if unassigned)
    "source_region",   # integer region index from .house (-1 if unassigned)
]


# =============================================================================
# Bearing / angle helpers
# =============================================================================

def parse_points(points_file):
    """Parse points.txt into {index: (x, y, z)}."""
    positions = {}
    with open(points_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                idx = int(parts[0])
                positions[idx] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return positions


def compute_bearing(receiver_pos, source_pos):
    """
    Horizontal bearing from receiver to source in Z-up space, degrees [0, 360).
    heading=0 is the -Y direction; bearing=90 is +X (right).
    Returns None for same-XY self-pairs.
    """
    dx = source_pos[0] - receiver_pos[0]
    dy = source_pos[1] - receiver_pos[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dx, -dy)) % 360


def select_best_angle(bearing_deg):
    best_angle, min_diff = 0, float("inf")
    for angle in ANGLES:
        diff = abs(((bearing_deg - angle) + 180) % 360 - 180)
        if diff < min_diff:
            min_diff, best_angle = diff, angle
    return best_angle


def select_best_channel(bearing_deg, best_angle):
    delta = ((bearing_deg - best_angle) + 180) % 360 - 180
    return 1 if delta > 0 else 0


# =============================================================================
# Region assignment from .house file
# =============================================================================

def _point_in_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test. polygon is a list of (px, py) tuples."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def parse_house_regions(house_file):
    """
    Extract floor polygon vertices and Z range per region from a .house file.

    Parses:
      R lines  → region_idx, ceiling height (parts[15])
      S lines  → floor surface index per region (label 'F')
      V lines  → polygon (x, y) vertices and z coordinates per floor surface

    Returns {region_idx: {'polygon': [(x, y), ...], 'z_min': float, 'H_m': float}}
    for regions with ≥ 3 floor vertices and a valid positive ceiling height.
    """
    region_height     = {}   # region_idx -> H_m from R line
    floor_surf_to_rgn = {}   # surface_idx -> region_idx
    surf_xy           = {}   # surface_idx -> [(x, y)]
    surf_z            = {}   # surface_idx -> [z]

    with open(house_file) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'R' and len(parts) >= 16:
                ridx = int(parts[1])
                region_height[ridx] = float(parts[15])
            elif parts[0] == 'S' and len(parts) >= 5 and parts[4] == 'F':
                surf_idx = int(parts[1])
                ridx     = int(parts[2])
                floor_surf_to_rgn[surf_idx] = ridx
                surf_xy.setdefault(surf_idx, [])
                surf_z.setdefault(surf_idx, [])
            elif parts[0] == 'V' and len(parts) >= 7:
                surf_idx = int(parts[2])
                if surf_idx in floor_surf_to_rgn:
                    surf_xy[surf_idx].append((float(parts[4]), float(parts[5])))
                    surf_z[surf_idx].append(float(parts[6]))

    result = {}
    for surf_idx, ridx in floor_surf_to_rgn.items():
        verts = surf_xy.get(surf_idx, [])
        H_m   = region_height.get(ridx)
        if len(verts) < 3 or H_m is None or H_m <= 0:
            continue
        result[ridx] = {
            'polygon': verts,
            'z_min':   min(surf_z[surf_idx]),
            'H_m':     H_m,
        }

    return result


def assign_regions(positions, region_polygons):
    """
    Assign each navigable point to a region using point-in-polygon and Z range.

    Checks both the 2D XY floor polygon and the vertical range [z_min, z_min + H_m]
    to correctly handle multi-floor buildings where rooms on different floors share
    the same XY footprint. A 0.1 m margin is added on both ends to absorb floating-
    point imprecision in the floor vertex heights.

    Returns {point_idx: region_idx}, with -1 for unassigned points.
    """
    assignments = {}
    for pt_idx, (px, py, pz) in positions.items():
        assigned = -1
        for region_idx, region in region_polygons.items():
            z_lo = region['z_min'] - 0.1
            z_hi = region['z_min'] + region['H_m'] + 0.1
            if z_lo <= pz <= z_hi and _point_in_polygon(px, py, region['polygon']):
                assigned = region_idx
                break
        assignments[pt_idx] = assigned
    return assignments


# =============================================================================
# Per-scene processing
# =============================================================================

def process_scene(scene_name, metadata_root, rir_root, mesh_root):
    meta_dir = metadata_root / scene_name
    positions = parse_points(meta_dir / "points.txt")

    with open(meta_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    graph_nodes = sorted(graph.nodes())

    house_file = mesh_root / scene_name / f"{scene_name}.house"
    if house_file.exists():
        region_polygons    = parse_house_regions(house_file)
        region_assignments = assign_regions(positions, region_polygons)
    else:
        region_polygons    = {}
        region_assignments = {idx: -1 for idx in positions}

    rows = []
    n_self_pairs = 0
    n_valid      = 0
    n_invalid    = 0

    for receiver_idx in graph_nodes:
        for source_idx in graph_nodes:
            rx, ry, rz = positions[receiver_idx]
            sx, sy, sz = positions[source_idx]

            bearing = compute_bearing((rx, ry, rz), (sx, sy, sz))
            if bearing is None:
                n_self_pairs += 1
                continue

            best_angle   = select_best_angle(bearing)
            best_channel = select_best_channel(bearing, best_angle)

            rir_rel = f"{scene_name}/data/binaural_rirs/mp3d/{scene_name}/{best_angle}/{receiver_idx}_{source_idx}.wav"
            rir_abs = rir_root / rir_rel
            valid   = rir_abs.exists() and rir_abs.stat().st_size >= MIN_RIR_BYTES

            r_region    = region_assignments.get(receiver_idx, -1)
            s_region    = region_assignments.get(source_idx, -1)
            same_region = (r_region != -1) and (r_region == s_region)

            rows.append({
                "scene":           scene_name,
                "receiver_idx":    receiver_idx,
                "source_idx":      source_idx,
                "bearing_deg":     round(bearing, 4),
                "best_angle":      best_angle,
                "best_channel":    best_channel,
                "receiver_x": rx, "receiver_y": ry, "receiver_z": rz,
                "source_x":   sx, "source_y":   sy, "source_z":   sz,
                "distance_m":      round(math.dist((rx, ry, rz), (sx, sy, sz)), 4),
                "rir_path":        rir_rel,
                "valid":           valid,
                "same_region":     same_region,
                "receiver_region": r_region,
                "source_region":   s_region,
            })

            if valid:
                n_valid += 1
            else:
                n_invalid += 1

    n_same_region = sum(1 for r in rows if r["same_region"])
    return rows, {
        "scene":                scene_name,
        "n_graph_nodes":        len(graph_nodes),
        "n_pairs_written":      len(rows),
        "n_valid":              n_valid,
        "n_invalid":            n_invalid,
        "n_self_pairs_skipped": n_self_pairs,
        "n_same_region":        n_same_region,
        "n_regions":            len(region_polygons),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute mono-RIR mapping for SoundSpaces Matterport3D dataset."
    )
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--rir-root",      type=Path, default=DEFAULT_RIR_ROOT)
    parser.add_argument("--mesh-root",     type=Path, default=DEFAULT_MESH_ROOT,
                        help="Directory of extracted mp3d_habitat meshes (for .house files)")
    parser.add_argument("--output-dir",    type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenes",        type=str,  default=None,
                        help="Comma-separated scene names (default: all scenes in metadata root)")
    args = parser.parse_args()

    for attr, label in [("metadata_root", "metadata root"), ("rir_root", "RIR root")]:
        p = getattr(args, attr)
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            return

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]
    else:
        scenes = sorted(d.name for d in args.metadata_root.iterdir() if d.is_dir())

    n_missing = sum(1 for s in scenes if not (args.mesh_root / s / f"{s}.house").exists())
    if n_missing:
        print(f"WARNING: {n_missing}/{len(scenes)} scene(s) missing .house file — same_region=False for those.")

    print(f"Processing {len(scenes)} scene(s)...\n")
    all_rows, all_stats, errors = [], [], []
    t_start = time.time()

    for i, scene in enumerate(scenes, 1):
        print(f"  [{i}/{len(scenes)}] {scene}...", end=" ", flush=True)
        try:
            rows, stats = process_scene(scene, args.metadata_root, args.rir_root, args.mesh_root)
        except Exception as e:
            print(f"ERROR — {e}")
            errors.append((scene, str(e)))
            continue

        all_rows.extend(rows)
        all_stats.append(stats)

        elapsed   = time.time() - t_start
        avg       = elapsed / i
        remaining = avg * (len(scenes) - i)
        same_pct  = 100 * stats["n_same_region"] / stats["n_pairs_written"] if stats["n_pairs_written"] else 0
        print(
            f"{stats['n_valid']} valid / {stats['n_invalid']} invalid"
            f" | {stats['n_same_region']:,} same-region ({same_pct:.0f}%)"
            f" | {stats['n_regions']} regions"
            + (f" | {stats['n_self_pairs_skipped']} self-pairs skipped" if stats["n_self_pairs_skipped"] else "")
            + f"  |  elapsed {timedelta(seconds=int(elapsed))}  ETA {timedelta(seconds=int(remaining))}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem        = "soundspaces_matterport3d_mapping"
    suffix      = "_incomplete" if errors else ""
    output_file = args.output_dir / f"{stem}{suffix}.csv"

    # Keep the columns other scripts attached; a pair holds its identity across rebuilds.
    columns = CSV_COLUMNS + carry_over_columns(args.output_dir / f"{stem}.csv", all_rows,
                                               ['rt60', 'void_frac'])

    with open(output_file, "w", newline="") as f:
        f.write(f"# SoundSpaces Matterport3D — Mono-RIR Mapping\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Metadata root: {args.metadata_root}\n")
        f.write(f"# RIR root: {args.rir_root}\n")
        f.write(f"# Mesh root: {args.mesh_root}\n")
        if errors:
            f.write("#\n")
            f.write(f"# INCOMPLETE — {len(errors)} scene(s) failed and are missing from this file:\n")
            for scene_name, reason in errors:
                f.write(f"#   {scene_name}: {reason}\n")
        f.write("#\n")
        f.write("# Column descriptions:\n")
        f.write("#   receiver_idx   : listener position — first number in RIR filename\n")
        f.write("#   source_idx     : sound source position — second number in RIR filename\n")
        f.write("#   bearing_deg    : horizontal bearing receiver->source [0, 360)\n")
        f.write("#   best_angle     : SoundSpaces angle folder {0,90,180,270} closest to bearing\n")
        f.write("#   best_channel   : 0=left ear, 1=right ear (closer to source)\n")
        f.write("#   distance_m     : Euclidean 3D distance in meters\n")
        f.write("#   rir_path       : relative path within RIR root\n")
        f.write(f"#   valid          : True if the WAV exists and holds at least {MIN_RIR_BYTES} bytes\n")
        f.write("#   same_region    : True if both points fall in the same .house floor polygon\n")
        f.write("#   receiver_region: .house region index (-1 if unassigned)\n")
        f.write("#   source_region  : .house region index (-1 if unassigned)\n")
        f.write("#\n")
        f.write("# Navigable points sit at z ≈ 1.52 m (ear height above floor).\n")
        f.write("#\n")

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    total_pairs       = sum(s["n_pairs_written"] for s in all_stats)
    total_valid       = sum(s["n_valid"]         for s in all_stats)
    total_invalid     = sum(s["n_invalid"]       for s in all_stats)
    total_same_region = sum(s["n_same_region"]   for s in all_stats)

    print(f"\nTotal pairs: {total_pairs:,}  ({total_valid:,} valid, {total_invalid:,} invalid)")
    if total_pairs:
        print(f"Same-region: {total_same_region:,} ({100*total_same_region/total_pairs:.1f}% of all pairs)")
    print(f"Mapping saved to: {output_file}")

    if total_valid > 0:
        angle_counts   = {0: 0, 90: 0, 180: 0, 270: 0}
        channel_counts = {0: 0, 1: 0}
        for row in all_rows:
            if row["valid"]:
                angle_counts[row["best_angle"]] += 1
                channel_counts[row["best_channel"]] += 1
        print(f"\nAngle distribution (valid only, ~25% each expected):")
        for a, c in angle_counts.items():
            print(f"  angle={a:>3}: {c:>8,}  ({100*c/total_valid:.1f}%)")
        print(f"Channel distribution:")
        print(f"  left  (0): {channel_counts[0]:>8,}  ({100*channel_counts[0]/total_valid:.1f}%)")
        print(f"  right (1): {channel_counts[1]:>8,}  ({100*channel_counts[1]/total_valid:.1f}%)")

    write_scene_stats(pd.DataFrame(all_rows, columns=columns), 'matterport3d',
                      args.output_dir / "analysis" / "soundspaces_scene_stats.txt",
                      regions_per_scene={s['scene']: s['n_regions'] for s in all_stats})

    if errors:
        print(f"\n{'!'*60}")
        print(f"  WARNING: {len(errors)} scene(s) FAILED — output saved as:")
        print(f"  {output_file}")
        print(f"  Failed scenes:")
        for scene_name, reason in errors:
            print(f"    {scene_name}: {reason}")
        print(f"{'!'*60}")


if __name__ == "__main__":
    main()
