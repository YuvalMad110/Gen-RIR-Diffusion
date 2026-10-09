"""
mp3d_house.py
-------------
Shared geometry utilities for Matterport3D .house file parsing and camera placement.

Imported by:
  05_render_mp3d_overviews.py     — rendering
  05b_visualize_mp3d_overviews.py — debug floor-plan visualization
"""

from __future__ import annotations

import math
from pathlib import Path

CORNER_OFFSET_M = 0.2   # metres inward from each corner wall


def parse_house_levels(house_path: Path) -> dict:
    """Parse R-lines → {region_idx: level_idx}."""
    region_level: dict[int, int] = {}
    try:
        with open(house_path) as f:
            for line in f:
                parts = line.split()
                # R region_idx level_idx 0 0 label ...
                if parts and parts[0] == "R" and len(parts) >= 3:
                    region_level[int(parts[1])] = int(parts[2])
    except FileNotFoundError:
        pass
    return region_level


def parse_house_polygons(house_path: Path) -> dict:
    """
    Parse a .house file → {region_idx: [(x, y), ...]} floor polygon vertices.

    V-lines store surface_idx (not region_idx) in column 2.  S-lines provide
    the surface → region mapping and identify floor surfaces (label 'F').
    """
    floor_surf_to_rgn: dict[int, int] = {}
    surf_vertices:     dict[int, list] = {}
    try:
        with open(house_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                tag = parts[0]
                # S surface_idx region_idx 0 label ...
                if tag == "S" and len(parts) >= 5 and parts[4] == "F":
                    sidx, ridx = int(parts[1]), int(parts[2])
                    floor_surf_to_rgn[sidx] = ridx
                    surf_vertices.setdefault(sidx, [])
                # V vertex_idx surface_idx label x y z ...
                elif tag == "V" and len(parts) >= 7:
                    sidx = int(parts[2])
                    if sidx in floor_surf_to_rgn:
                        surf_vertices[sidx].append((float(parts[4]), float(parts[5])))
    except FileNotFoundError:
        pass

    polygons: dict[int, list] = {}
    for sidx, ridx in floor_surf_to_rgn.items():
        verts = surf_vertices.get(sidx, [])
        if verts:
            polygons.setdefault(ridx, []).extend(verts)
    return polygons


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Ray-casting point-in-polygon test (2D)."""
    n = len(polygon)
    inside = False
    xi, yi = polygon[0]
    for j in range(1, n + 1):
        xj, yj = polygon[j % n]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        xi, yi = xj, yj
    return inside


def closest_point_on_segment(px: float, py: float,
                              ax: float, ay: float,
                              bx: float, by: float) -> tuple[float, float]:
    """Closest point on segment AB to point P (2D)."""
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    return ax + t * dx, ay + t * dy


def closest_point_on_polygon(px: float, py: float,
                              polygon: list) -> tuple[float, float]:
    """Closest point on the polygon perimeter to point P (2D)."""
    best_d2 = float("inf")
    best_x, best_y = polygon[0]
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        cx, cy = closest_point_on_segment(px, py, ax, ay, bx, by)
        d2 = (cx - px) ** 2 + (cy - py) ** 2
        if d2 < best_d2:
            best_d2, best_x, best_y = d2, cx, cy
    return best_x, best_y
