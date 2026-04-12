"""
04_render_replica_poses.py
---------------------------
Renders per-pair images from Replica 3D scenes using ReplicaPoseRenderer (C++ binary).
Camera is placed at the receiver position facing the source — one image per (receiver, source)
pair in the SoundSpaces mapping CSV.

Output folders:
  target2source_rgb/{scene}/tgt{r}_src{s}.jpg    -- RGB 1280x960 JPEG
  target2source_depth/{scene}/tgt{r}_src{s}.png  -- 16-bit depth PNG

Test mode (--max-pairs set):
  Output goes to ~/temp/Replica_rendered/ to keep test data separate from production.

Usage:
    # Quick test (10 pairs, office_1, RGB only)
    python3 data/data_preparation/04_render_replica_poses.py \\
        --scenes office_1 --max-pairs 10 --types rgb

    # Full render, all scenes, RGB + depth
    python3 data/data_preparation/04_render_replica_poses.py --types rgb,depth

Requirements:
    ReplicaPoseRenderer binary built at:
    ~/tools/Replica-Dataset/build/ReplicaSDK/ReplicaPoseRenderer
    (run tools/setup_replica_renderer.sh to build it)
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# ─── Default paths ────────────────────────────────────────────────────────────

DEFAULT_REPLICA_ROOT  = "/dsi/gannot-lab/gannot-lab1/datasets/Replica"
DEFAULT_OUTPUT_ROOT   = "/dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered"
TEST_OUTPUT_ROOT      = str(Path.home() / "temp" / "Replica_rendered")
DEFAULT_MAPPING_CSV   = Path(__file__).resolve().parents[1] / "soundspaces_replica_mapping.csv"
DEFAULT_BINARY        = Path.home() / "tools" / "Replica-Dataset" / "build" / "ReplicaSDK" / "ReplicaPoseRenderer"
DEFAULT_POSES_DIR     = Path(__file__).resolve().parents[1] / "poses"
EGL_ENV_VAR           = "__EGL_VENDOR_LIBRARY_FILENAMES"
EGL_VENDOR_FILE       = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# ─── CSV loading ──────────────────────────────────────────────────────────────

def load_mapping(csv_path: Path, scene: str, max_pairs: int = None) -> list:
    """Load rows for a scene from the mapping CSV. Returns list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            if row["scene"] != scene:
                continue
            rows.append(row)
            if max_pairs and len(rows) >= max_pairs:
                break
    return rows


# ─── Pose CSV generation ──────────────────────────────────────────────────────

def write_poses_csv(rows: list, poses_path: Path, skip_existing: bool,
                    output_dir_rgb: Path, output_dir_depth: Path, types: list):
    """
    Write a poses CSV for ReplicaPoseRenderer.
    Lines that already have output files are omitted (unless skip_existing=False).

    Format:  eye_x eye_y eye_z  tgt_x tgt_y tgt_z  stem
    """
    poses_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with open(poses_path, "w") as f:
        f.write("# eye_x eye_y eye_z  tgt_x tgt_y tgt_z  stem\n")
        for row in rows:
            r = int(row["receiver_idx"])
            s = int(row["source_idx"])
            stem = f"tgt{r}_src{s}"

            # Skip if all requested output files already exist
            if skip_existing:
                already_done = all([
                    (output_dir_rgb   / f"{stem}.jpg").exists() if "rgb"   in types else True,
                    (output_dir_depth / f"{stem}.png").exists() if "depth" in types else True,
                ])
                if already_done:
                    skipped += 1
                    continue

            ex = float(row["receiver_x"])
            ey = float(row["receiver_y"])
            ez = float(row["receiver_z"])
            tx = float(row["source_x"])
            ty = float(row["source_y"])
            tz = float(row["source_z"])
            f.write(f"{ex:.6f} {ey:.6f} {ez:.6f}  {tx:.6f} {ty:.6f} {tz:.6f}  {stem}\n")
            written += 1

    return written, skipped


# ─── Rendering ────────────────────────────────────────────────────────────────

def run_renderer(binary: Path, mesh: Path, atlas: Path, poses_csv: Path,
                 out_dir: Path, sensor_type: str, glass: Path = None):
    """
    Call ReplicaPoseRenderer for one (scene, type) combination.
    sensor_type: 'rgb' or 'depth'
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    flag = "--rgb-only" if sensor_type == "rgb" else "--depth-only"

    cmd = [str(binary), str(mesh), str(atlas), str(poses_csv), str(out_dir), flag]
    if glass and glass.exists():
        cmd.append(str(glass))

    env = os.environ.copy()
    env[EGL_ENV_VAR] = EGL_VENDOR_FILE

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n  ERROR: renderer exited with code {result.returncode}")
        sys.exit(result.returncode)


# ─── Per-scene entry point ────────────────────────────────────────────────────

def process_scene(scene: str, args, output_root: Path, all_types: list):
    replica_scene = Path(args.replica_root) / scene
    mesh   = replica_scene / "mesh.ply"
    atlas  = replica_scene / "textures"
    glass  = replica_scene / "glass.sur"

    if not mesh.exists():
        print(f"  WARNING: mesh.ply not found for scene '{scene}', skipping.")
        return

    # Load pairs from mapping CSV
    rows = load_mapping(args.mapping_csv, scene, args.max_pairs)
    if not rows:
        print(f"  WARNING: no rows for scene '{scene}' in mapping CSV, skipping.")
        return

    print(f"  {len(rows)} pairs to render...")

    # Output dirs per type
    out_rgb   = output_root / "target2source_rgb"   / scene
    out_depth = output_root / "target2source_depth" / scene

    # Write poses CSV (filtering already-rendered pairs)
    poses_path = Path(args.poses_dir) / f"{scene}.csv"
    written, skipped = write_poses_csv(
        rows, poses_path,
        skip_existing=not args.no_skip,
        output_dir_rgb=out_rgb,
        output_dir_depth=out_depth,
        types=all_types,
    )

    if written == 0:
        print(f"  All pairs already rendered, skipping.")
        return

    if skipped:
        print(f"  Skipped {skipped} already-rendered pairs, rendering {written} new.")

    # Render each requested type
    for sensor_type in all_types:
        out_dir = out_rgb if sensor_type == "rgb" else out_depth
        print(f"  Rendering {sensor_type}...", flush=True)
        run_renderer(Path(args.binary), mesh, atlas, poses_path, out_dir, sensor_type, glass)
        n_out = len(list(out_dir.glob("tgt*.jpg" if sensor_type == "rgb" else "tgt*.png")))
        print(f"    → {n_out} files in {out_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render per-pair Replica images using ReplicaPoseRenderer."
    )
    parser.add_argument("--replica-root",  default=DEFAULT_REPLICA_ROOT,
                        help="Root of Replica dataset")
    parser.add_argument("--output-root",   default=DEFAULT_OUTPUT_ROOT,
                        help="Production output root (overridden by ~/temp/... when --max-pairs is set)")
    parser.add_argument("--mapping-csv",   type=Path, default=DEFAULT_MAPPING_CSV,
                        help="Path to soundspaces_replica_mapping.csv")
    parser.add_argument("--binary",        default=str(DEFAULT_BINARY),
                        help="Path to ReplicaPoseRenderer binary")
    parser.add_argument("--scenes",        default=None,
                        help="Comma-separated scene names (default: all from mapping CSV)")
    parser.add_argument("--max-pairs",     type=int, default=None,
                        help="Max pairs per scene — triggers test output path ~/temp/Replica_rendered/")
    parser.add_argument("--types",         default="rgb",
                        help="Comma-separated: rgb,depth  (default: rgb)")
    parser.add_argument("--poses-dir",     default=str(DEFAULT_POSES_DIR),
                        help="Directory for temporary per-scene pose CSVs")
    parser.add_argument("--no-skip",       action="store_true",
                        help="Re-render even if output file already exists")
    args = parser.parse_args()

    # Validate binary
    if not Path(args.binary).exists():
        print(f"ERROR: ReplicaPoseRenderer binary not found: {args.binary}")
        print("Run tools/setup_replica_renderer.sh to build it.")
        sys.exit(1)

    # Validate EGL vendor file
    if not Path(EGL_VENDOR_FILE).exists():
        print(f"WARNING: EGL vendor file not found: {EGL_VENDOR_FILE}")
        print("NVIDIA EGL may not initialise correctly.")

    # Parse types
    types = [t.strip() for t in args.types.split(",")]
    for t in types:
        if t not in ("rgb", "depth"):
            print(f"ERROR: unknown type '{t}'. Choose from: rgb, depth")
            sys.exit(1)

    # Output root: explicit --output-root wins; otherwise test mode uses ~/temp/Replica_rendered/
    if args.output_root != DEFAULT_OUTPUT_ROOT:
        output_root = Path(args.output_root)
    elif args.max_pairs is not None:
        output_root = Path(TEST_OUTPUT_ROOT)
        print(f"Test mode (--max-pairs {args.max_pairs}): output → {output_root}")
    else:
        output_root = Path(args.output_root)

    # Discover scenes
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]
    else:
        scenes = []
        with open(args.mapping_csv, newline="") as f:
            reader = csv.DictReader(line for line in f if not line.startswith("#"))
            for row in reader:
                if row["scene"] not in scenes:
                    scenes.append(row["scene"])

    print(f"\nRendering {len(scenes)} scene(s)  |  types: {types}")
    print(f"Output root: {output_root}\n")

    for scene in scenes:
        print(f"Scene: {scene}")
        process_scene(scene, args, output_root, types)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
