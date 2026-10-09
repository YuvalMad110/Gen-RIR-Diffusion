"""
05b_visualize_mp3d_overviews.py
---------------------------------
Draws a top-down floor plan for one MP3D scene showing:
  - Room floor polygons (from .house file V lines)
  - Room bounding boxes (oriented rectangles from room_geometry.csv)
  - Room center mark (×) with region ID label
  - Overview camera positions (numbered dots, one set per room)
  - Look-at arrows (horizontal projection of each camera's gaze toward room center)

Camera placement uses the same logic as compute_room_cameras in
05_render_mp3d_overviews.py via shared helpers from mp3d_house.py.

Usage
-----
    # default: scene 17DRP5sb8fy
    python3 data/data_preparation/05b_visualize_mp3d_overviews.py

    # specify scene
    python3 data/data_preparation/05b_visualize_mp3d_overviews.py --scene 17DRP5sb8fy

    # custom paths
    python3 data/data_preparation/05b_visualize_mp3d_overviews.py \\
        --scene 17DRP5sb8fy \\
        --geometry-csv data/matterport3d/room_geometry.csv \\
        --mesh-root /dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/meshes \\
        --out-dir data/matterport3d/room_geometry_vis/floor_vis
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mp3d_house import (CORNER_OFFSET_M, parse_house_levels, parse_house_polygons,
                        point_in_polygon, closest_point_on_polygon)

# ── Visualization parameters ────────────────────────────────────────────────
ARROW_LEN   = 0.55   # metres — look-at arrow length in floor plan
DOT_SIZE    = 50     # scatter dot size for camera positions
CENTER_SIZE = 60     # scatter dot size for room centers

DEFAULT_GEOMETRY_CSV = (
    Path(__file__).resolve().parents[1] / "matterport3d" / "room_geometry.csv"
)
DEFAULT_MESH_ROOT = Path("/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/meshes")
DEFAULT_OUT_DIR   = (
    Path(__file__).resolve().parents[1] / "matterport3d" / "room_geometry_vis" / "floor_vis"
)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def bbox_corners(cx, cy, L, W, theta_rad):
    """Return the 4 corners of an oriented bounding box as [(x,y), ...] in loop order."""
    corners = []
    for us, vs in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
        u, v = us * L / 2, vs * W / 2
        wx = cx + u * math.cos(theta_rad) - v * math.sin(theta_rad)
        wy = cy + u * math.sin(theta_rad) + v * math.cos(theta_rad)
        corners.append((wx, wy))
    return corners


def room_cameras(cx, cy, L, W, theta_rad, polygon=None):
    """
    Return [(corner_num, world_x, world_y), ...] — mirrors compute_room_cameras
    in 05_render_mp3d_overviews.py including the polygon-snap fallback.
    """
    half_u = max(L / 2 - CORNER_OFFSET_M, 0.0)
    half_v = max(W / 2 - CORNER_OFFSET_M, 0.0)

    use_polygon = polygon is not None and len(polygon) >= 3
    if use_polygon:
        poly_cx = sum(p[0] for p in polygon) / len(polygon)
        poly_cy = sum(p[1] for p in polygon) / len(polygon)

    cams = []
    for num, us, vs in [(1, 1, 1), (2, 1, -1), (3, -1, 1), (4, -1, -1)]:
        u, v = us * half_u, vs * half_v
        wx = cx + u * math.cos(theta_rad) - v * math.sin(theta_rad)
        wy = cy + u * math.sin(theta_rad) + v * math.cos(theta_rad)

        if use_polygon and not point_in_polygon(wx, wy, polygon):
            bx, by = closest_point_on_polygon(wx, wy, polygon)
            dx, dy = poly_cx - bx, poly_cy - by
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                wx = bx + dx / dist * CORNER_OFFSET_M
                wy = by + dy / dist * CORNER_OFFSET_M
            else:
                wx, wy = bx, by

        cams.append((num, wx, wy))
    return cams


# ── Visualization ────────────────────────────────────────────────────────────

def _draw_floor(ax, rooms, polygons, colors):
    """Draw one floor's rooms onto ax."""
    for room in rooms:
        ridx  = int(room["region_idx"])
        cx    = float(room["center_x"])
        cy    = float(room["center_y"])
        L     = float(room["L_m"])
        W     = float(room["W_m"])
        theta = math.radians(float(room["rotation_deg"]))
        color = colors[ridx % len(colors)]

        # Floor polygon
        if ridx in polygons:
            pts = np.array(polygons[ridx])
            xs = np.append(pts[:, 0], pts[0, 0])
            ys = np.append(pts[:, 1], pts[0, 1])
            ax.fill(pts[:, 0], pts[:, 1], alpha=0.18, color=color, zorder=1)
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.7, zorder=2)

        # Oriented bounding box
        corners = bbox_corners(cx, cy, L, W, theta)
        bxs = [c[0] for c in corners] + [corners[0][0]]
        bys = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(bxs, bys, color=color, linewidth=1.5, linestyle="--",
                alpha=0.85, zorder=3)

        # Room center
        ax.scatter([cx], [cy], color=color, s=CENTER_SIZE, marker="x",
                   linewidths=2.5, zorder=5)
        ax.annotate(f"r{ridx:03d}", (cx, cy),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8.5, color=color, fontweight="bold", zorder=6)

        # Camera positions and look-at arrows
        for num, wcx, wcy in room_cameras(cx, cy, L, W, theta, polygon=polygons.get(ridx)):
            ax.scatter([wcx], [wcy], s=DOT_SIZE, color=color, marker="o",
                       edgecolors="black", linewidths=0.7, zorder=7)
            ax.annotate(str(num), (wcx, wcy),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7.5, color="black", zorder=8)

            dx, dy = cx - wcx, cy - wcy
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                dx, dy = dx / dist * ARROW_LEN, dy / dist * ARROW_LEN
            ax.annotate(
                "", xy=(wcx + dx, wcy + dy), xytext=(wcx, wcy),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.6, mutation_scale=12),
                zorder=7,
            )


def visualize_scene(scene: str, geometry_csv: Path, mesh_root: Path, out_dir: Path):
    rooms = []
    with open(geometry_csv, newline="") as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            if row["scene"] == scene:
                rooms.append(row)
    if not rooms:
        print(f"No rooms found for scene {scene} in {geometry_csv}")
        return

    house_path = mesh_root / scene / f"{scene}.house"
    polygons     = parse_house_polygons(house_path)
    region_level = parse_house_levels(house_path)
    if not polygons:
        print(f"  (no floor polygons found — {house_path})")

    # Group rooms by floor level
    rooms_by_level: dict[int, list] = {}
    for room in rooms:
        level = region_level.get(int(room["region_idx"]), 0)
        rooms_by_level.setdefault(level, []).append(room)

    n_levels = len(rooms_by_level)
    colors   = plt.cm.tab20.colors
    out_dir.mkdir(parents=True, exist_ok=True)

    for level_idx, level_rooms in sorted(rooms_by_level.items()):
        fig, ax = plt.subplots(figsize=(22, 15))
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f0f0")
        fig.patch.set_facecolor("#ffffff")

        _draw_floor(ax, level_rooms, polygons, colors)

        ax.set_xlabel("World X (m)", fontsize=11)
        ax.set_ylabel("World Y (m)", fontsize=11)
        floor_label = f"floor {level_idx}" if n_levels > 1 else "single floor"
        ax.set_title(
            f"MP3D scene {scene}  —  {floor_label}  ({len(level_rooms)} rooms)  "
            f"room overview camera positions\n"
            f"dashed = bounding box  │  filled = floor polygon  │  × = center  "
            f"│  ● = camera  │  → = look-at",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

        floor_suffix = f"_floor{level_idx}" if n_levels > 1 else ""
        out_path = out_dir / f"{scene}{floor_suffix}_overview_floor_plan.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MP3D room overview camera positions as a top-down floor plan."
    )
    parser.add_argument("--scene",        default="17DRP5sb8fy",
                        help="Scene name (default: 17DRP5sb8fy)")
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument("--mesh-root",    type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--out-dir",      type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    for scene in [s.strip() for s in args.scene.split(",")]:
        visualize_scene(scene, args.geometry_csv, args.mesh_root, args.out_dir)


if __name__ == "__main__":
    main()
