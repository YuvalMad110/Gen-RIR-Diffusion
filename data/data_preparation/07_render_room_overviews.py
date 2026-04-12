"""
07_render_room_overviews.py
----------------------------
Renders 8 overview images per Replica scene — one from each corner of the mesh bounding box.

The 8 corners are the 4 XY corners × 2 Z levels:
  top_corner*    — CAMERA_OFFSET metres above the ceiling, looking down at the floor center
  bottom_corner* — CAMERA_OFFSET metres below the floor, looking up at the ceiling center

Corner numbering (XY): 1=(x_min,y_min)  2=(x_min,y_max)  3=(x_max,y_min)  4=(x_max,y_max)

XY corners come from the mesh bounding box.
Z values (floor = z_min, ceiling = z_min + H_m) come from room_geometry.csv.

Output:
  room_overviews/{scene}/top_corner{1-4}_rgb.jpg
  room_overviews/{scene}/bottom_corner{1-4}_rgb.jpg
  (and _depth.png equivalents when --types includes depth)

Usage:
    python3 data/data_preparation/07_render_room_overviews.py
    python3 data/data_preparation/07_render_room_overviews.py --scenes office_2,room_0
    python3 data/data_preparation/07_render_room_overviews.py --scenes office_2 --types rgb,depth
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
DEFAULT_OUTPUT_ROOT  = "/dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered"
TEST_OUTPUT_ROOT     = str(Path.home() / "temp" / "Replica_rendered")
DEFAULT_GEOMETRY_CSV = Path(__file__).resolve().parents[1] / "room_geometry.csv"
DEFAULT_BINARY       = Path.home() / "tools" / "Replica-Dataset" / "build" / "ReplicaSDK" / "ReplicaPoseRenderer"
DEFAULT_POSES_DIR    = Path(__file__).resolve().parents[1] / "poses"
EGL_ENV_VAR          = "__EGL_VENDOR_LIBRARY_FILENAMES"
EGL_VENDOR_FILE      = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

VERTEX_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
])

# Offset from ceiling/floor surface for camera placement (metres).
CAMERA_OFFSET = 0.5

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

def compute_corner_poses(x_min, x_max, y_min, y_max, z_min, z_max,
                         center_x, center_y):
    """
    Return 8 (stem_prefix, eye, target) tuples — one per bounding-box corner.

    stem_prefix does not include the sensor-type suffix; callers append '_rgb' or '_depth'.
    4 top poses look down at the floor center; 4 bottom poses look up at the ceiling center.
    """
    xy_corners = [
        (1, x_min, y_min),
        (2, x_min, y_max),
        (3, x_max, y_min),
        (4, x_max, y_max),
    ]

    poses = []
    for num, cx, cy in xy_corners:
        poses.append((f"top_corner{num}", (cx, cy, z_max + CAMERA_OFFSET),
                      (center_x, center_y, z_min)))
        poses.append((f"bottom_corner{num}", (cx, cy, z_min - CAMERA_OFFSET),
                      (center_x, center_y, z_max)))

    return poses


# ─── Rendering ────────────────────────────────────────────────────────────────

def run_renderer(binary: Path, mesh: Path, atlas: Path, poses_csv: Path,
                 out_dir: Path, sensor_type: str, glass: Path = None):
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

def process_scene(scene: str, geom: dict, args, output_root: Path, types: list):
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

    poses = compute_corner_poses(
        x_min, x_max, y_min, y_max, z_min, z_max, center_x, center_y)

    Path(args.poses_dir).mkdir(parents=True, exist_ok=True)
    out_dir = output_root / "room_overviews" / scene

    for sensor_type in types:
        ext = "jpg" if sensor_type == "rgb" else "png"
        if not args.no_skip:
            all_exist = all(
                (out_dir / f"{prefix}_{sensor_type}.{ext}").exists()
                for prefix, _, _ in poses
            )
            if all_exist:
                print(f"  [{sensor_type}] all 8 images already rendered, skipping.")
                continue

        poses_path = Path(args.poses_dir) / f"{scene}_overviews_{sensor_type}.csv"
        with open(poses_path, "w") as f:
            f.write("# eye_x eye_y eye_z  tgt_x tgt_y tgt_z  stem\n")
            for prefix, (ex, ey, ez), (tx, ty, tz) in poses:
                f.write(f"{ex:.6f} {ey:.6f} {ez:.6f}  "
                        f"{tx:.6f} {ty:.6f} {tz:.6f}  {prefix}_{sensor_type}\n")

        print(f"  Rendering {sensor_type}...", flush=True)
        run_renderer(DEFAULT_BINARY, mesh, atlas, poses_path, out_dir, sensor_type, glass)
        print(f"  → {out_dir}/"
              f"  [{', '.join(f'{p}_{sensor_type}.{ext}' for p, _, _ in poses)}]")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default=None,
                        help="Comma-separated list of scenes (default: all 18)")
    parser.add_argument("--types", default="rgb",
                        help="Comma-separated: rgb, depth (default: rgb)")
    parser.add_argument("--replica-root", default=DEFAULT_REPLICA_ROOT)
    parser.add_argument("--output-root", default=None,
                        help="Output root (default: test path if --scenes given, else production)")
    parser.add_argument("--geometry-csv", default=str(DEFAULT_GEOMETRY_CSV))
    parser.add_argument("--poses-dir", default=str(DEFAULT_POSES_DIR))
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-render even if output files already exist")
    args = parser.parse_args()

    scenes = args.scenes.split(",") if args.scenes else ALL_SCENES
    types  = args.types.split(",")

    if args.output_root:
        output_root = Path(args.output_root)
    elif args.scenes:
        output_root = Path(TEST_OUTPUT_ROOT)
    else:
        output_root = Path(DEFAULT_OUTPUT_ROOT)

    geom_by_scene = {}
    with open(args.geometry_csv, newline="") as f:
        for row in csv.DictReader(f):
            geom_by_scene[row["scene"]] = row

    print(f"Rendering room overview images  |  {len(scenes)} scene(s)  |  types: {types}")
    print(f"Output root: {output_root}\n")

    for scene in scenes:
        print(f"Scene: {scene}")
        if scene not in geom_by_scene:
            print(f"  WARNING: no geometry entry for '{scene}', skipping.")
            continue
        process_scene(scene, geom_by_scene[scene], args, output_root, types)

    print("\nDone.")


if __name__ == "__main__":
    main()
