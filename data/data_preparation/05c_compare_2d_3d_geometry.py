"""
05c_compare_2d_3d_geometry.py
------------------------------
Diagnostic overlay: compare 2D .house room geometry with 3D data to detect
coordinate alignment between the .house file and the .glb mesh.

For each scene, produces one PNG per floor with four layers:
  - Room polygons (filled, semi-transparent) and bounding boxes (dashed)  — from .house / room_geometry.csv
  - Navigable points from points.txt (green circles)                       — SoundSpaces mic/speaker grid
  - Mesh floor vertices, Y-up interpretation: (X, Z) where Y ≈ min(Y)     — orange dots
  - Mesh floor vertices, Z-up interpretation: (X, Y) where Z ≈ z_min      — purple dots

If the orange cloud aligns with the polygons → the .glb uses Y-up (X,Z horizontal).
If the purple cloud aligns              → the .glb uses Z-up (X,Y horizontal).

Usage (run from host, CPU-only — no GPU needed):
    /home/dsi/yuvalmad/.conda/envs/habitat/bin/python3 \\
        /home/dsi/yuvalmad/Projects/Gen-RIR-Diffusion/data/data_preparation/05c_compare_2d_3d_geometry.py \\
        --scenes 17DRP5sb8fy

    # Multiple scenes:
    ... --scenes 17DRP5sb8fy,1LXtFkjw3qL
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mp3d_house import (CORNER_OFFSET_M, closest_point_on_polygon, parse_house_levels,
                        parse_house_polygons, point_in_polygon)

# ── Default paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_GEOMETRY_CSV = _SCRIPT_DIR.parent / "matterport3d" / "room_geometry.csv"
DEFAULT_MESH_ROOT    = Path("/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/meshes")
DEFAULT_NAV_ROOT     = Path("/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/metadata/mp3d")
DEFAULT_OUT_DIR      = _SCRIPT_DIR.parent / "matterport3d" / "room_geometry_vis" / "coord_test"

# Height of the SoundSpaces sensor above the floor (metres)
_SENSOR_HEIGHT_M = 1.52
# Tolerance for matching a nav point's z to a floor level (±metres)
_NAV_Z_TOLERANCE = 0.5
# Floor vertex filter: include mesh vertices within this many metres of the floor height
_FLOOR_VERT_THRESH = 0.3


# ── Data loading ───────────────────────────────────────────────────────────────

def load_rooms_csv(geometry_csv: Path, scene: str) -> list[dict]:
    rows: list[dict] = []
    with open(geometry_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["scene"] == scene:
                rows.append(row)
    return rows


def load_nav_points(nav_root: Path, scene: str) -> np.ndarray | None:
    """Return (N, 3) float array of (x, y, z) from points.txt, or None."""
    pts_path = nav_root / scene / "points.txt"
    if not pts_path.exists():
        print(f"  [warn] points.txt not found: {pts_path}")
        return None
    pts: list[tuple[float, float, float]] = []
    with open(pts_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                pts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.array(pts, dtype=float) if pts else None


def load_mesh_floor_vertices(
    glb_path: Path,
    z_floor: float,
) -> dict[str, np.ndarray] | None:
    """
    Load all vertices from the .glb mesh and return two candidate floor-vertex sets.

    Y-up interpretation: floor is near the global Y minimum; horizontal axes are (X, Z).
    Z-up interpretation: floor is near z_floor (from .house R-line); horizontal axes are (X, Y).

    Returns {'yup': (N,2), 'zup': (N,2)} or None if trimesh is unavailable / load fails.
    """
    try:
        import trimesh
    except ImportError:
        print("  [warn] trimesh not available — skipping mesh vertex overlay")
        return None

    try:
        obj = trimesh.load(str(glb_path), force="scene")
        if isinstance(obj, trimesh.Scene):
            meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                return None
            verts = np.vstack([m.vertices for m in meshes])
        elif isinstance(obj, trimesh.Trimesh):
            verts = obj.vertices
        else:
            return None
    except Exception as e:
        print(f"  [warn] trimesh failed to load {glb_path.name}: {e}")
        return None

    result: dict[str, np.ndarray] = {}

    y_min = float(verts[:, 1].min())
    mask_yup = verts[:, 1] < y_min + _FLOOR_VERT_THRESH
    if mask_yup.sum() > 0:
        result["yup"] = verts[mask_yup][:, [0, 2]]   # (X, Z)

    mask_zup = np.abs(verts[:, 2] - z_floor) < _FLOOR_VERT_THRESH
    if mask_zup.sum() > 0:
        result["zup"] = verts[mask_zup][:, [0, 1]]   # (X, Y)

    return result or None


# ── Drawing ────────────────────────────────────────────────────────────────────

def room_cameras(room: dict, polygon: list) -> list:
    """
    The four overview camera positions and their look-at target, in world X/Y.

    Mirrors the placement in 05_render_mp3d_overviews.py: each camera sits
    CORNER_OFFSET_M inward from the two nearest corner walls, snapped onto the floor
    polygon when the bounding-box corner falls outside it.
    """
    cx, cy = float(room["center_x"]), float(room["center_y"])
    L, W   = float(room["L_m"]), float(room["W_m"])
    theta  = math.radians(float(room["rotation_deg"]))
    half_u = max(L / 2 - CORNER_OFFSET_M, 0.0)
    half_v = max(W / 2 - CORNER_OFFSET_M, 0.0)

    use_poly = polygon is not None and len(polygon) >= 3
    if use_poly:
        poly_cx = sum(p[0] for p in polygon) / len(polygon)
        poly_cy = sum(p[1] for p in polygon) / len(polygon)

    cams = []
    for num, u_sign, v_sign in [(1, +1, +1), (2, +1, -1), (3, -1, +1), (4, -1, -1)]:
        u, v = u_sign * half_u, v_sign * half_v
        wx = cx + u * math.cos(theta) - v * math.sin(theta)
        wy = cy + u * math.sin(theta) + v * math.cos(theta)
        snapped = False
        if use_poly and not point_in_polygon(wx, wy, polygon):
            snapped = True
            bx, by = closest_point_on_polygon(wx, wy, polygon)
            dx, dy = poly_cx - bx, poly_cy - by
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                wx = bx + dx / dist * CORNER_OFFSET_M
                wy = by + dy / dist * CORNER_OFFSET_M
            else:
                wx, wy = bx, by
        cams.append((num, wx, wy, cx, cy, snapped))
    return cams


def draw_cameras(ax, rooms: list[dict], polygons: dict, colors) -> None:
    """Camera dots, index labels and look-at arrows, one set per room."""
    for room in rooms:
        ridx  = int(room["region_idx"])
        color = colors[ridx % len(colors)]
        for num, wx, wy, tx, ty, snapped in room_cameras(room, polygons.get(ridx)):
            ax.scatter([wx], [wy], s=46, color=color, zorder=10,
                       edgecolors="black" if snapped else "white", linewidths=1.1)
            ax.annotate(f"{num}", (wx, wy), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color="black", zorder=11)
            dx, dy = tx - wx, ty - wy
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                ax.arrow(wx, wy, dx / norm * 0.7, dy / norm * 0.7,
                         head_width=0.16, head_length=0.2, fc=color, ec=color,
                         alpha=0.85, zorder=9, length_includes_head=True)


def draw_rooms(ax, rooms: list[dict], polygons: dict, colors) -> None:
    for room in rooms:
        ridx  = int(room["region_idx"])
        cx    = float(room["center_x"])
        cy    = float(room["center_y"])
        L     = float(room["L_m"])
        W     = float(room["W_m"])
        theta = math.radians(float(room["rotation_deg"]))
        color = colors[ridx % len(colors)]

        if ridx in polygons:
            pts = np.array(polygons[ridx])
            closed_x = np.append(pts[:, 0], pts[0, 0])
            closed_y = np.append(pts[:, 1], pts[0, 1])
            ax.fill(pts[:, 0], pts[:, 1], alpha=0.15, color=color, zorder=1)
            ax.plot(closed_x, closed_y, color=color, lw=1.5, alpha=0.8, zorder=2)

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        hw, hh = L / 2, W / 2
        corners = [
            (cx + u * cos_t - v * sin_t, cy + u * sin_t + v * cos_t)
            for u, v in [(hw, hh), (hw, -hh), (-hw, -hh), (-hw, hh)]
        ]
        bxs = [c[0] for c in corners] + [corners[0][0]]
        bys = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(bxs, bys, color=color, lw=1.2, ls="--", alpha=0.7, zorder=3)

        ax.annotate(
            f"r{ridx:03d}", (cx, cy),
            ha="center", va="center",
            fontsize=8, color=color, fontweight="bold", zorder=6,
        )


# ── Per-scene entry point ──────────────────────────────────────────────────────

def visualize_scene(
    scene: str,
    geometry_csv: Path,
    mesh_root: Path,
    nav_root: Path,
    out_dir: Path,
) -> None:
    rooms = load_rooms_csv(geometry_csv, scene)
    if not rooms:
        print(f"  [warn] no rows in CSV for scene '{scene}'")
        return

    house_path = mesh_root / scene / f"{scene}.house"
    glb_path   = mesh_root / scene / f"{scene}.glb"

    polygons     = parse_house_polygons(house_path)
    region_level = parse_house_levels(house_path)

    nav_pts = load_nav_points(nav_root, scene)

    # Group rooms by floor level
    rooms_by_level: dict[int, list] = {}
    for room in rooms:
        lvl = region_level.get(int(room["region_idx"]), 0)
        rooms_by_level.setdefault(lvl, []).append(room)

    n_levels = len(rooms_by_level)
    colors   = plt.cm.tab20.colors
    out_dir.mkdir(parents=True, exist_ok=True)

    for level_idx, level_rooms in sorted(rooms_by_level.items()):
        z_floor = float(np.mean([float(r["z_min"]) for r in level_rooms]))

        mesh_verts = load_mesh_floor_vertices(glb_path, z_floor)

        fig, ax = plt.subplots(figsize=(22, 15))
        ax.set_aspect("equal")
        ax.set_facecolor("#f5f5f5")
        fig.patch.set_facecolor("#ffffff")

        draw_rooms(ax, level_rooms, polygons, colors)
        draw_cameras(ax, level_rooms, polygons, colors)

        # Nav points
        if nav_pts is not None:
            z_sensor = z_floor + _SENSOR_HEIGHT_M
            mask = np.abs(nav_pts[:, 2] - z_sensor) < _NAV_Z_TOLERANCE
            floor_nav = nav_pts[mask]
            if len(floor_nav) > 0:
                ax.scatter(
                    floor_nav[:, 0], floor_nav[:, 1],
                    c="limegreen", s=16, alpha=0.85, zorder=8,
                    label=f"Nav pts (points.txt)  n={len(floor_nav)}",
                )

        # Mesh floor vertices — two axis interpretations
        if mesh_verts:
            if "yup" in mesh_verts:
                n = len(mesh_verts["yup"])
                ax.scatter(
                    mesh_verts["yup"][:, 0], mesh_verts["yup"][:, 1],
                    c="darkorange", s=1, alpha=0.2, zorder=7,
                    label=f"Mesh Y-up floor  (mesh_X, mesh_Z)  n={n}",
                )
            if "zup" in mesh_verts:
                n = len(mesh_verts["zup"])
                ax.scatter(
                    mesh_verts["zup"][:, 0], mesh_verts["zup"][:, 1],
                    c="mediumpurple", s=1, alpha=0.2, zorder=7,
                    label=f"Mesh Z-up floor  (mesh_X, mesh_Y)  n={n}",
                )

        floor_tag = f"floor {level_idx}" if n_levels > 1 else "single floor"
        ax.set_title(
            f"MP3D {scene}  —  {floor_tag}  —  2D / 3D coordinate alignment check\n"
            "filled polygon + dashed bbox = .house   │   "
            "green ● = nav pts (points.txt)   │   "
            "orange ● = mesh Y-up (X,Z)   │   purple ● = mesh Z-up (X,Y)\n"
            "numbered ● = overview camera (black edge = polygon-snapped)   │   → = look-at",
            fontsize=11,
        )
        ax.set_xlabel("World X (m)", fontsize=11)
        ax.set_ylabel("World Y (m)", fontsize=11)
        ax.legend(loc="upper right", fontsize=9, markerscale=4)
        ax.grid(True, alpha=0.3, lw=0.5)

        suffix   = f"_floor{level_idx}" if n_levels > 1 else ""
        out_path = out_dir / f"{scene}{suffix}_coord_alignment.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay 2D .house geometry with 3D nav points and mesh floor vertices.",
    )
    parser.add_argument(
        "--scenes", default="17DRP5sb8fy",
        help="Comma-separated scene IDs  (default: 17DRP5sb8fy)",
    )
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV,
                        help="Path to room_geometry.csv")
    parser.add_argument("--mesh-root",    type=Path, default=DEFAULT_MESH_ROOT,
                        help="Root dir containing one subfolder per scene (*.glb, *.house)")
    parser.add_argument("--nav-root",     type=Path, default=DEFAULT_NAV_ROOT,
                        help="SoundSpaces metadata root (one subfolder per scene with points.txt)")
    parser.add_argument("--out-dir",      type=Path, default=DEFAULT_OUT_DIR,
                        help="Output directory for PNG files")
    args = parser.parse_args()

    for scene in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        print(f"\n=== {scene} ===")
        visualize_scene(
            scene,
            args.geometry_csv,
            args.mesh_root,
            args.nav_root,
            args.out_dir,
        )


if __name__ == "__main__":
    main()
