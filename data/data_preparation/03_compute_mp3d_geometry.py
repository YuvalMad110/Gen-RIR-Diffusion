"""
03_compute_mp3d_geometry.py
----------------------------
Computes the physical geometry of each room (region) in every Matterport3D scene
from its .house file.

For each (scene, region):
  1. Reads the floor-surface polygon vertices (V lines with S-label 'F').
  2. Computes the 2D convex hull of the room footprint.
  3. Finds the minimum-area bounding rectangle → L, W, rotation angle θ.
  4. z_min  = floor vertex z (precise per-room floor height).
  5. H_m    = 'height' field from the R line (per-room ceiling height).

Regions whose floor surface has fewer than 3 vertices are skipped (degenerate /
transitional spaces such as doorways or closets). Those pairs are already
excluded by same_region=False in the mapping CSV (points in degenerate regions
get receiver_region=-1).

Outputs:
  data/matterport3d/room_geometry.csv    — one row per (scene, region_idx)
  data/matterport3d/room_geometry_vis/   — one PNG per room

CSV columns match data/replica/room_geometry.csv except for the added
'region_idx' key. SoundSpacesMatterportDataset looks up geometry by
(scene, region_idx) using receiver_region from the mapping CSV.

Usage:
    python3 data/data_preparation/03_compute_mp3d_geometry.py
    python3 data/data_preparation/03_compute_mp3d_geometry.py --scenes 17DRP5sb8fy
    python3 data/data_preparation/03_compute_mp3d_geometry.py --no-viz
"""

import argparse
import csv
import math
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull

from scene_stats import GEOMETRY_KEY, carry_over_columns

DEFAULT_MESH_ROOT  = "/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/meshes"
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parents[1] / "matterport3d" / "room_geometry.csv"
DEFAULT_VIZ_DIR    = Path(__file__).resolve().parents[1] / "matterport3d" / "room_geometry_vis"

# ---------------------------------------------------------------------------
# Column definitions
#
# scene        : Matterport3D scene name (e.g. "17DRP5sb8fy").
# region_idx   : Room index from the .house file — unique within the scene.
#                (scene, region_idx) is the globally unique room key.
#                Joins to receiver_region / source_region in the mapping CSV.
#
# label        : Matterport region category from the R line ('bedroom', 'porch',
#                'hallway', ...). OUTDOOR_LABELS marks the open-to-sky categories,
#                which behave differently from enclosed rooms both acoustically and
#                in how much of a render is empty space.
#
# outdoor      : True when label is one of OUTDOOR_LABELS. The dataset reads this to
#                pick the indoor or the outdoor void threshold for a room.
#
# L_m          : Long side of the minimum-area bounding rectangle (metres).
# W_m          : Short side of the minimum-area bounding rectangle (metres).
# H_m          : Floor-to-ceiling height for this room (metres), taken from the
#                'height' field (parts[15]) of the R line in the .house file.
#
# center_x     : X coordinate of the bounding-rectangle centre in world (scene)
# center_y       coordinates — the same system as receiver_x / receiver_y in the
#                mapping CSV.  NOT aligned with L or W.
#
# rotation_deg : Angle of the long axis (L) from world +X, CCW positive,
#                in [0°, 180°).  L and W are NOT aligned with the world X/Y axes
#                in general; you need this angle to convert between world and
#                room-aligned coordinates.
#
#                World → room-aligned transform for a point (x, y, z):
#                  dx, dy = x - center_x,  y - center_y
#                  θ      = radians(rotation_deg)
#                  u      =  dx*cos(θ) + dy*sin(θ)   # along L, in [-L/2, L/2]
#                  v      = -dx*sin(θ) + dy*cos(θ)   # along W, in [-W/2, W/2]
#                  z_norm = z - z_min                 # height above floor (≈1.52 m
#                                                     # for any navigable point)
#
# z_min        : World-Z of the room's floor, taken from the V (vertex) lines of
#                the floor surface in the .house file.  The ceiling is at world
#                Z ≈ z_min + H_m.  Navigable points sit at z ≈ z_min + 1.52 m
#                (ear / sensor height).
#
# hull_area_m2 : 2D convex-hull area of the floor polygon (m²) — the actual
#                room footprint.  Always ≤ rect_area_m2.
# rect_area_m2 : Bounding-rectangle area L × W (m²).  The ratio
#                hull_area_m2 / rect_area_m2 measures how rectangular the room
#                is; a ratio of 1.0 indicates a perfect rectangle.
# ---------------------------------------------------------------------------

# Matterport region category codes from the R-line label field.
REGION_LABELS = {
    'a': 'bathroom',  'b': 'bedroom',  'c': 'closet',   'd': 'dining',
    'e': 'entry',     'f': 'family',   'g': 'garage',   'h': 'hallway',
    'i': 'library',   'j': 'laundry',  'k': 'kitchen',  'l': 'living',
    'm': 'meeting',   'n': 'lounge',   'o': 'office',   'p': 'porch',
    'r': 'rec',       's': 'stairs',   't': 'toilet',   'u': 'utility',
    'v': 'tvroom',    'w': 'workout',  'x': 'outdoor',  'y': 'balcony',
    'z': 'other',     'B': 'bar',      'C': 'classroom','D': 'dining_booth',
    'S': 'spa',       'Z': 'junk',     '-': 'none',
}

# Open-to-sky categories; the rest are enclosed.
OUTDOOR_LABELS = {'porch', 'outdoor', 'balcony'}

CSV_COLUMNS = [
    "scene",
    "region_idx",
    "label",
    "outdoor",
    "L_m",
    "W_m",
    "H_m",
    "center_x",
    "center_y",
    "rotation_deg",
    "z_min",
    "hull_area_m2",
    "rect_area_m2",
]


# =============================================================================
# .house file parsing
# =============================================================================

def parse_house_rooms(house_file):
    """
    Extract per-room geometry from a .house file.

    Parses:
      R lines  → region_idx, ceiling height (parts[15])
      S lines  → floor surface index per region (label 'F')
      V lines  → polygon vertices per floor surface

    Returns a list of dicts, one per region with a valid floor polygon (≥ 3 vertices):
        {
          'region_idx': int,
          'xy':         np.ndarray (N, 2)  — floor polygon XY vertices,
          'z_min':      float              — floor vertex z,
          'H_m':        float              — ceiling height from R line,
        }

    Regions with fewer than 3 floor vertices or a non-positive ceiling height
    are omitted (degenerate spaces).
    """
    region_height      = {}   # region_idx -> float
    region_label       = {}   # region_idx -> str
    floor_surf_to_rgn  = {}   # surface_idx -> region_idx
    surf_xy            = {}   # surface_idx -> [(x, y)]
    surf_z             = {}   # surface_idx -> [z]

    with open(house_file) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]

            # R region_idx level_idx 0 0 label px py pz xlo ylo zlo xhi yhi zhi height ...
            if tag == 'R' and len(parts) >= 16:
                ridx   = int(parts[1])
                height = float(parts[15])
                region_height[ridx] = height
                region_label[ridx]  = REGION_LABELS.get(parts[5], parts[5])

            # S surface_idx region_idx 0 label ...
            elif tag == 'S' and len(parts) >= 5 and parts[4] == 'F':
                sidx = int(parts[1])
                ridx = int(parts[2])
                floor_surf_to_rgn[sidx] = ridx
                surf_xy.setdefault(sidx, [])
                surf_z.setdefault(sidx, [])

            # V vertex_idx surface_idx label px py pz ...
            elif tag == 'V' and len(parts) >= 7:
                sidx = int(parts[2])
                if sidx in floor_surf_to_rgn:
                    surf_xy[sidx].append((float(parts[4]), float(parts[5])))
                    surf_z[sidx].append(float(parts[6]))

    rooms = []
    for sidx, ridx in floor_surf_to_rgn.items():
        verts  = surf_xy.get(sidx, [])
        height = region_height.get(ridx)
        if len(verts) < 3 or height is None or height <= 0:
            continue
        rooms.append({
            'region_idx': ridx,
            'label':      region_label.get(ridx, 'none'),
            'xy':         np.array(verts, dtype=float),
            'z_min':      float(min(surf_z[sidx])),
            'H_m':        height,
        })

    return rooms


# =============================================================================
# Geometry — identical algorithm to 06_compute_room_geometry.py
# =============================================================================

def min_area_bounding_rect(points_2d):
    """
    Minimum-area bounding rectangle of a 2D point cloud via rotating calipers.

    Returns:
        angle_deg  — long-axis angle from +X (degrees, CCW positive, in [0, 180))
        L          — long side (metres)
        W          — short side (metres)
        center     — (cx, cy) of rectangle centre
        corners    — (4, 2) rectangle corners
        hull       — scipy ConvexHull (hull.volume = 2D area)
    """
    hull     = ConvexHull(points_2d)
    hull_pts = points_2d[hull.vertices]

    best_area  = np.inf
    best_angle = 0.0
    best_rect  = None

    n = len(hull_pts)
    for i in range(n):
        edge  = hull_pts[(i + 1) % n] - hull_pts[i]
        angle = math.atan2(edge[1], edge[0])
        c, s  = math.cos(-angle), math.sin(-angle)
        rot   = np.array([[c, -s], [s, c]])

        rotated = hull_pts @ rot.T
        mn, mx  = rotated.min(axis=0), rotated.max(axis=0)
        dims    = mx - mn
        area    = dims[0] * dims[1]

        if area < best_area:
            best_area  = area
            best_angle = angle
            ctr_rot    = (mn + mx) / 2
            c2, s2     = math.cos(angle), math.sin(angle)
            rot_inv    = np.array([[c2, -s2], [s2, c2]])
            center     = ctr_rot @ rot_inv.T
            corners_r  = np.array([
                [mn[0], mn[1]], [mx[0], mn[1]],
                [mx[0], mx[1]], [mn[0], mx[1]],
            ])
            corners    = corners_r @ rot_inv.T
            best_rect  = (dims[0], dims[1], center, corners)

    dim0, dim1, center, corners = best_rect
    if dim0 >= dim1:
        L, W      = dim0, dim1
        angle_deg = math.degrees(best_angle)
    else:
        L, W      = dim1, dim0
        angle_deg = math.degrees(best_angle + math.pi / 2)

    angle_deg = angle_deg % 180
    return angle_deg, L, W, center, corners, hull


# =============================================================================
# Visualisation
# =============================================================================

def save_viz(scene, region_idx, xy, hull, corners, center, L, W, angle_deg, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(xy[:, 0], xy[:, 1], s=6, c="lightgrey", alpha=0.8)

    hull_pts    = xy[hull.vertices]
    hull_closed = np.vstack([hull_pts, hull_pts[0]])
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], "b-", lw=1.5)

    rect_closed = np.vstack([corners, corners[0]])
    ax.plot(rect_closed[:, 0], rect_closed[:, 1], "r-", lw=2)

    θ = math.radians(angle_deg)
    ax.plot(*center, "r+", ms=8, mew=2)
    ax.annotate("", xy=(center[0] + L / 2 * math.cos(θ),
                        center[1] + L / 2 * math.sin(θ)),
                xytext=center,
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))

    ax.set_aspect("equal")
    ax.set_title(f"{scene} / region {region_idx}  |  "
                 f"L={L:.2f} m  W={W:.2f} m  θ={angle_deg:.1f}°", fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


# =============================================================================
# Per-scene processing
# =============================================================================

def process_scene(scene, mesh_root, viz_dir, make_viz):
    """
    Compute per-room geometry for one scene. Returns a list of CSV row dicts.
    """
    house_file = Path(mesh_root) / scene / f"{scene}.house"
    if not house_file.exists():
        raise FileNotFoundError(f".house not found: {house_file}")

    rooms = parse_house_rooms(house_file)
    rows  = []

    for room in rooms:
        ridx = room['region_idx']
        xy   = room['xy']
        try:
            angle_deg, L, W, center, corners, hull = min_area_bounding_rect(xy)
        except Exception as e:
            # ConvexHull fails on collinear points (e.g. a thin corridor reduced to a line)
            print(f"    region {ridx}: skipped — {e}")
            continue

        rows.append({
            "scene":        scene,
            "region_idx":   ridx,
            "label":        room['label'],
            "outdoor":      room['label'] in OUTDOOR_LABELS,
            "L_m":          round(L, 4),
            "W_m":          round(W, 4),
            "H_m":          round(room['H_m'], 4),
            "center_x":     round(float(center[0]), 6),
            "center_y":     round(float(center[1]), 6),
            "rotation_deg": round(angle_deg, 4),
            "z_min":        round(room['z_min'], 6),
            "hull_area_m2": round(float(hull.volume), 4),
            "rect_area_m2": round(L * W, 4),
        })

        if make_viz:
            viz_path = viz_dir / f"{scene}_r{ridx:03d}.png"
            save_viz(scene, ridx, xy, hull, corners, center, L, W, angle_deg, viz_path)

    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-room geometry for MP3D scenes from .house files."
    )
    parser.add_argument("--mesh-root",  type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--viz-dir",    type=Path, default=DEFAULT_VIZ_DIR)
    parser.add_argument("--scenes",     type=str,  default=None,
                        help="Comma-separated scene names (default: all in mesh-root)")
    parser.add_argument("--no-viz",     action="store_true")
    args = parser.parse_args()

    if not args.mesh_root.exists():
        print(f"ERROR: mesh root not found: {args.mesh_root}")
        return

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]
    else:
        scenes = sorted(
            d.name for d in args.mesh_root.iterdir()
            if d.is_dir() and (d / f"{d.name}.house").exists()
        )

    print(f"Processing {len(scenes)} scene(s)  |  per-room geometry\n")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        args.viz_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    errors   = []
    t_start  = time.time()

    for i, scene in enumerate(scenes, 1):
        try:
            rows = process_scene(scene, args.mesh_root, args.viz_dir, not args.no_viz)
        except FileNotFoundError as e:
            print(f"\n  [{i}/{len(scenes)}] {scene}: ERROR — {e}")
            errors.append((scene, str(e)))
            continue

        all_rows.extend(rows)

        elapsed   = time.time() - t_start
        avg       = elapsed / i
        remaining = avg * (len(scenes) - i)
        print(f"  [{i}/{len(scenes)}] {scene}: {len(rows)} rooms"
              f"  |  elapsed {timedelta(seconds=int(elapsed))}"
              f"  |  ETA {timedelta(seconds=int(remaining))}")

    # Choose output path — _incomplete suffix flags that some scenes are missing.
    if errors:
        output_csv = args.output_csv.with_stem(args.output_csv.stem + "_incomplete")
    else:
        output_csv = args.output_csv

    # Keep the columns other scripts attached; a room holds its identity across rebuilds.
    columns = CSV_COLUMNS + carry_over_columns(output_csv, all_rows,
                                               ['mean_overview_void'], key=GEOMETRY_KEY)

    with open(output_csv, "w", newline="") as f:
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Mesh root: {args.mesh_root}\n")
        f.write(f"# OUTDOOR_LABELS: {', '.join(sorted(OUTDOOR_LABELS))}\n")
        f.write("#\n")
        if errors:
            f.write(f"# INCOMPLETE — {len(errors)} scene(s) failed and are missing from this file:\n")
            for scene_name, reason in errors:
                f.write(f"#   {scene_name}: {reason}\n")
            f.write("#\n")
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    if errors:
        print(f"\n{'!'*60}")
        print(f"  WARNING: {len(errors)} scene(s) FAILED — output saved as:")
        print(f"  {output_csv}")
        print(f"  Failed scenes:")
        for scene_name, reason in errors:
            print(f"    {scene_name}: {reason}")
        print(f"{'!'*60}")
    else:
        print(f"\nWrote {len(all_rows)} room rows from {len(scenes)} scenes → {output_csv}")
    if not args.no_viz:
        print(f"Visualisations in {args.viz_dir}")


if __name__ == "__main__":
    main()
