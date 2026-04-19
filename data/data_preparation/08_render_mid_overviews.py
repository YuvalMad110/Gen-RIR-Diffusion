"""
08_render_mid_overviews.py
---------------------------
Renders 4 mid-height overview images per Replica scene — one from each corner of the mesh
bounding box, with the camera at the vertical centre of the room (z_min + H_m/2) looking
horizontally toward the room centre.

Corner numbering (XY): 1=(x_min,y_min)  2=(x_min,y_max)  3=(x_max,y_min)  4=(x_max,y_max)

Output (default: ~/temp/mid_overview_test/):
  {scene}/mid_corner{1-4}_rgb.jpg

Usage:
    python3 data/data_preparation/08_render_mid_overviews.py
    python3 data/data_preparation/08_render_mid_overviews.py --scenes office_2,room_0
    python3 data/data_preparation/08_render_mid_overviews.py --scenes office_2 --output-root ~/temp/mid_overview_test
"""

import argparse
import csv
import os
import subprocess
import sys
import numpy as np
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_REPLICA_ROOT = "/dsi/gannot-lab/gannot-lab1/datasets/Replica"
DEFAULT_OUTPUT_ROOT  = str(Path.home() / "temp" / "mid_overview_test")
DEFAULT_GEOMETRY_CSV = Path(__file__).resolve().parents[1] / "room_geometry.csv"
DEFAULT_BINARY       = Path.home() / "tools" / "Replica-Dataset" / "build" / "ReplicaSDK" / "ReplicaPoseRenderer"
EGL_ENV_VAR          = "__EGL_VENDOR_LIBRARY_FILENAMES"
EGL_VENDOR_FILE      = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

VERTEX_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
])

ALL_SCENES = [
    "apartment_0", "apartment_1", "apartment_2",
    "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
    "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
    "hotel_0",
    "office_0", "office_1", "office_2", "office_3", "office_4",
    "room_0", "room_1", "room_2",
]


# ─── Mesh XY bounding box ─────────────────────────────────────────────────────

def mesh_xy_bounds(ply_path: Path):
    """Return (x_min, x_max, y_min, y_max) from mesh.ply."""
    n_verts = 0
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n_verts = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        verts = np.frombuffer(f.read(n_verts * VERTEX_DTYPE.itemsize), dtype=VERTEX_DTYPE)

    return (float(verts["x"].min()), float(verts["x"].max()),
            float(verts["y"].min()), float(verts["y"].max()))


# ─── Pose computation ─────────────────────────────────────────────────────────

def compute_mid_poses(x_min, x_max, y_min, y_max, z_min, z_max, center_x, center_y):
    """
    Return 4 (stem_prefix, eye, target) tuples — one per bounding-box corner.

    Camera placed at the vertical centre of the room (z_min + H/2), looking horizontally
    toward the room centre at the same height.
    """
    mid_z = z_min + (z_max - z_min) / 2.0
    xy_corners = [
        (1, x_min, y_min),
        (2, x_min, y_max),
        (3, x_max, y_min),
        (4, x_max, y_max),
    ]
    poses = []
    for num, cx, cy in xy_corners:
        poses.append((f"mid_corner{num}", (cx, cy, mid_z), (center_x, center_y, mid_z)))
    return poses


# ─── Rendering ────────────────────────────────────────────────────────────────

def run_renderer(binary: Path, mesh: Path, atlas: Path, poses_csv: Path,
                 out_dir: Path, glass: Path = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), str(mesh), str(atlas), str(poses_csv), str(out_dir), "--rgb-only"]
    if glass and glass.exists():
        cmd.append(str(glass))
    env = os.environ.copy()
    env[EGL_ENV_VAR] = EGL_VENDOR_FILE
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n  ERROR: renderer exited with code {result.returncode}")
        sys.exit(result.returncode)


# ─── Per-scene entry point ────────────────────────────────────────────────────

def process_scene(scene: str, geom: dict, args, output_root: Path):
    replica_scene = Path(args.replica_root) / scene
    mesh  = replica_scene / "mesh.ply"
    atlas = replica_scene / "textures"
    glass = replica_scene / "glass.sur"

    if not mesh.exists():
        print(f"  WARNING: mesh.ply not found, skipping.")
        return

    print(f"  Reading mesh XY bounds...", end=" ", flush=True)
    x_min, x_max, y_min, y_max = mesh_xy_bounds(mesh)
    print(f"done  ([{x_min:.2f},{x_max:.2f}] × [{y_min:.2f},{y_max:.2f}])")

    center_x = float(geom["center_x"])
    center_y = float(geom["center_y"])
    z_min    = float(geom["z_min"])
    z_max    = z_min + float(geom["H_m"])

    poses = compute_mid_poses(x_min, x_max, y_min, y_max, z_min, z_max, center_x, center_y)

    out_dir   = output_root / scene
    poses_dir = output_root / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_skip:
        all_exist = all((out_dir / f"{prefix}_rgb.jpg").exists() for prefix, _, _ in poses)
        if all_exist:
            print(f"  All 4 mid images already rendered, skipping.")
            return

    poses_path = poses_dir / f"{scene}_mid_rgb.csv"
    with open(poses_path, "w") as f:
        f.write("# eye_x eye_y eye_z  tgt_x tgt_y tgt_z  stem\n")
        for prefix, (ex, ey, ez), (tx, ty, tz) in poses:
            f.write(f"{ex:.6f} {ey:.6f} {ez:.6f}  "
                    f"{tx:.6f} {ty:.6f} {tz:.6f}  {prefix}_rgb\n")

    print(f"  Rendering mid RGB...", flush=True)
    run_renderer(DEFAULT_BINARY, mesh, atlas, poses_path, out_dir, glass)
    print(f"  → {out_dir}/  [{', '.join(f'{p}_rgb.jpg' for p, _, _ in poses)}]")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default=None,
                        help="Comma-separated list of scenes (default: all 18)")
    parser.add_argument("--replica-root", default=DEFAULT_REPLICA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--geometry-csv", default=str(DEFAULT_GEOMETRY_CSV))
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-render even if output files already exist")
    args = parser.parse_args()

    scenes      = args.scenes.split(",") if args.scenes else ALL_SCENES
    output_root = Path(args.output_root)

    geom_by_scene = {}
    with open(args.geometry_csv, newline="") as f:
        for row in csv.DictReader(f):
            geom_by_scene[row["scene"]] = row

    print(f"Rendering mid-height overview images  |  {len(scenes)} scene(s)")
    print(f"Output root: {output_root}\n")

    for scene in scenes:
        print(f"Scene: {scene}")
        if scene not in geom_by_scene:
            print(f"  WARNING: no geometry entry for '{scene}', skipping.")
            continue
        process_scene(scene, geom_by_scene[scene], args, output_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
