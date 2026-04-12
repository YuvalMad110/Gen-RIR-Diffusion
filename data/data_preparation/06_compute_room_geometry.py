"""
06_compute_room_geometry.py
----------------------------
Computes the physical geometry of each Replica scene from its mesh.ply file.

For each scene:
  1. Isolates floor-level vertices (z near z_min of the mesh)
  2. Computes the 2D convex hull of the floor footprint
  3. Finds the minimum-area bounding rectangle (rotating calipers) → L, W, rotation angle θ
  4. Computes room height H = z_max − z_min (ceiling to floor span)

Outputs:
  data/room_geometry.csv          — one row per scene
  data/replica_room_geometry/     — one PNG per scene (hull + bounding rect overlay)

The rotation angle and center in room_geometry.csv can be used to transform
receiver/source pts coordinates to room-aligned axes:
  dx = pts_x - center_x,  dy = pts_y - center_y
  u =  dx*cos(θ) + dy*sin(θ)   [metres along long axis]
  v = -dx*sin(θ) + dy*cos(θ)   [metres along short axis]

Usage:
    python3 data/data_preparation/06_compute_room_geometry.py
    python3 data/data_preparation/06_compute_room_geometry.py --scenes office_2,room_0
"""

import argparse
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DEFAULT_REPLICA_ROOT = "/dsi/gannot-lab/gannot-lab1/datasets/Replica"
PROJECT_ROOT         = Path(__file__).resolve().parents[2]
OUTPUT_CSV           = PROJECT_ROOT / "data" / "room_geometry.csv"
OUTPUT_VIZ_DIR       = PROJECT_ROOT / "data" / "replica_room_geometry"

ALL_SCENES = [
    "apartment_0", "apartment_1", "apartment_2",
    "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
    "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
    "hotel_0",
    "office_0", "office_1", "office_2", "office_3", "office_4",
    "room_0", "room_1", "room_2",
]

VERTEX_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
])

CSV_COLUMNS = [
    "scene",
    "L_m",           # long side of min-area bounding rectangle (metres)
    "W_m",           # short side of min-area bounding rectangle (metres)
    "H_m",           # floor-to-ceiling height (metres)
    "center_x",      # room center in pts/mesh XY coords
    "center_y",
    "rotation_deg",  # angle of long axis from +X axis (degrees), CCW positive
    "z_min",         # mesh z of the floor surface (pts/mesh coords)
    "hull_area_m2",  # convex hull area (m²)
    "rect_area_m2",  # bounding rectangle area (m²)
]


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_vertices(ply_path: Path) -> np.ndarray:
    """Load all vertices from a binary-little-endian PLY file."""
    n_verts = 0
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n_verts = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        return np.frombuffer(f.read(n_verts * VERTEX_DTYPE.itemsize), dtype=VERTEX_DTYPE)


def get_floor_vertices(verts: np.ndarray) -> np.ndarray:
    """
    Return the floor-level vertices for room footprint computation.

    Floor = lowest Z in the Z-up mesh. Take all vertices within 0.1m of z_min.
    """
    z_min = float(verts["z"].min())
    return verts[verts["z"] < z_min + 0.1]


# ── Geometry ──────────────────────────────────────────────────────────────────

def min_area_bounding_rect(points_2d: np.ndarray):
    """
    Find the minimum-area bounding rectangle of a 2D point cloud.

    Returns:
        angle_deg  — rotation of the long axis from +X (degrees, CCW positive)
        L          — length of long side (metres)
        W          — length of short side (metres)
        center     — (cx, cy) of the rectangle
        corners    — (4, 2) array of rectangle corners
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points_2d)
    hull_pts = points_2d[hull.vertices]

    best_area = np.inf
    best_angle = 0.0
    best_rect = None

    n = len(hull_pts)
    for i in range(n):
        # Edge direction
        edge = hull_pts[(i + 1) % n] - hull_pts[i]
        angle = math.atan2(edge[1], edge[0])
        c, s = math.cos(-angle), math.sin(-angle)
        rot = np.array([[c, -s], [s, c]])

        rotated = hull_pts @ rot.T
        mn, mx = rotated.min(axis=0), rotated.max(axis=0)
        dims = mx - mn
        area = dims[0] * dims[1]

        if area < best_area:
            best_area = area
            best_angle = angle
            # Rectangle center in rotated space → back to original
            ctr_rot = (mn + mx) / 2
            c2, s2 = math.cos(angle), math.sin(angle)
            rot_inv = np.array([[c2, -s2], [s2, c2]])
            center = ctr_rot @ rot_inv.T
            # Corners
            corners_rot = np.array([
                [mn[0], mn[1]], [mx[0], mn[1]],
                [mx[0], mx[1]], [mn[0], mx[1]],
            ])
            corners = corners_rot @ rot_inv.T
            best_rect = (dims[0], dims[1], center, corners)

    dim0, dim1, center, corners = best_rect
    # Ensure L >= W
    if dim0 >= dim1:
        L, W = dim0, dim1
        angle_deg = math.degrees(best_angle)
    else:
        L, W = dim1, dim0
        angle_deg = math.degrees(best_angle + math.pi / 2)

    # Normalise angle to [0, 180)
    angle_deg = angle_deg % 180

    return angle_deg, L, W, center, corners, hull


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_viz(scene: str, floor_pts: np.ndarray, hull, corners: np.ndarray,
             center, L: float, W: float, angle_deg: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Floor point cloud (thin scatter)
    ax.scatter(floor_pts[:, 0], floor_pts[:, 1], s=0.5, c="lightgrey", alpha=0.4)

    # Convex hull boundary
    hull_pts = floor_pts[hull.vertices]
    hull_closed = np.vstack([hull_pts, hull_pts[0]])
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], "b-", lw=1.5, label="Convex hull")

    # Bounding rectangle
    rect_closed = np.vstack([corners, corners[0]])
    ax.plot(rect_closed[:, 0], rect_closed[:, 1], "r-", lw=2, label="Min-area rect")

    # Center + long-axis arrow
    θ = math.radians(angle_deg)
    ax.plot(*center, "r+", ms=10, mew=2)
    ax.annotate("", xy=(center[0] + L / 2 * math.cos(θ), center[1] + L / 2 * math.sin(θ)),
                xytext=center,
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax.annotate("", xy=(center[0] + W / 2 * math.cos(θ + math.pi / 2),
                        center[1] + W / 2 * math.sin(θ + math.pi / 2)),
                xytext=center,
                arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2))

    ax.set_aspect("equal")
    ax.set_title(f"{scene}  |  L={L:.2f}m  W={W:.2f}m  θ={angle_deg:.1f}°")
    ax.set_xlabel("X (mesh / pts)")
    ax.set_ylabel("Y (mesh / pts)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ── Per-scene processing ──────────────────────────────────────────────────────

def process_scene(scene: str, replica_root: str) -> dict:
    ply_path = Path(replica_root) / scene / "mesh.ply"
    print(f"  Loading mesh ({ply_path.stat().st_size // 1_000_000} MB)...", end=" ", flush=True)
    verts = load_vertices(ply_path)
    print("done")

    z_min = float(verts["z"].min())
    floor_verts = get_floor_vertices(verts)
    floor_z = float(np.median(floor_verts["z"]))
    print(f"  Floor vertices: {len(floor_verts):,} (floor_z≈{floor_z:.4f})")

    floor_xy = np.column_stack([floor_verts["x"], floor_verts["y"]])

    # Min-area bounding rectangle
    angle_deg, L, W, center, corners, hull = min_area_bounding_rect(floor_xy)

    hull_pts = floor_xy[hull.vertices]
    hull_area = hull.volume  # scipy ConvexHull: .volume = area in 2D

    rect_area = L * W

    H = float(verts["z"].max()) - floor_z

    print(f"  L={L:.3f}m  W={W:.3f}m  H={H:.3f}m  θ={angle_deg:.2f}°  "
          f"hull={hull_area:.2f}m²  rect={rect_area:.2f}m²")

    return {
        "scene": scene,
        "L_m": round(L, 4),
        "W_m": round(W, 4),
        "H_m": round(H, 4),
        "center_x": round(float(center[0]), 6),
        "center_y": round(float(center[1]), 6),
        "rotation_deg": round(angle_deg, 4),
        "z_min": round(z_min, 6),
        "hull_area_m2": round(hull_area, 4),
        "rect_area_m2": round(rect_area, 4),
        # For viz
        "_floor_xy": floor_xy,
        "_hull": hull,
        "_corners": corners,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default=None,
                        help="Comma-separated list of scenes (default: all 18)")
    parser.add_argument("--replica-root", default=DEFAULT_REPLICA_ROOT)
    args = parser.parse_args()

    scenes = args.scenes.split(",") if args.scenes else ALL_SCENES
    OUTPUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in scenes:
        print(f"\n{scene}")
        row = process_scene(scene, args.replica_root)

        viz_path = OUTPUT_VIZ_DIR / f"{scene}.png"
        save_viz(scene, row["_floor_xy"], row["_hull"], row["_corners"],
                 (row["center_x"], row["center_y"]),
                 row["L_m"], row["W_m"], row["rotation_deg"], viz_path)
        print(f"  Viz saved: {viz_path}")

        rows.append({k: v for k, v in row.items() if not k.startswith("_")})

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"Verification PNGs in {OUTPUT_VIZ_DIR}")


if __name__ == "__main__":
    main()
