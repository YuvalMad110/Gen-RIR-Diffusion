"""
05_render_mp3d_overviews.py
----------------------------
Renders 4 room-overview RGB images per Matterport3D room (region) via Habitat-Sim.

Camera placement
----------------
Each room has 4 corner cameras defined in room-aligned coordinates, derived from
room_geometry.csv (center_x/y, L_m, W_m, rotation_deg, z_min, H_m):

  corner 1: (u=+L/2−0.2, v=+W/2−0.2)
  corner 2: (u=+L/2−0.2, v=−W/2+0.2)
  corner 3: (u=−L/2+0.2, v=+W/2−0.2)
  corner 4: (u=−L/2+0.2, v=−W/2+0.2)

Each camera is placed 0.2 m inward from the two nearest corner walls, giving a
consistent absolute clearance regardless of room size. Each camera sits at 85% of
ceiling height and looks toward the room center at 50% height. The look-at direction
includes a vertical component (full 3D quaternion, not just horizontal yaw).

Room → world transform (Z-up, θ = rotation_deg):
    world_x = center_x + u·cos(θ) − v·sin(θ)
    world_y = center_y + u·sin(θ) + v·cos(θ)
    world_z = z_min + fraction · H_m

Habitat Y-up transform: (world_x, world_z, −world_y).

Corner numbering is fixed: corner 1 is always at the (+u, +v) room-aligned corner.
This gives the dataset class a deterministic mapping from corner index to room-aligned
position — the model can learn spatial priors without any geometry reasoning.

Output
------
  {output_root}/room_overviews/{scene}_r{ridx:03d}/corner{1-4}_rgb.jpg  (224×294 JPEG)

Usage
-----
    conda activate habitat

    # smoke test — one scene
    python3 data/data_preparation/05_render_mp3d_overviews.py --scenes 17DRP5sb8fy

    # full render
    python3 data/data_preparation/05_render_mp3d_overviews.py
"""

from __future__ import annotations

import os
import sys

# glibc caches LD_LIBRARY_PATH at process startup — runtime changes via os.environ do
# not reach the dynamic linker for subsequent dlopen calls.  Re-exec with the right path
# so the fresh process sees it from the start.
#
# /host_lib64 is the RHEL host's /lib64 bind-mounted into the Ubuntu container
# (docker-compose.yml) to supply libEGL_nvidia.so.0 and its NVIDIA dependency chain.
# Ubuntu's own lib paths are prepended so Ubuntu's libc/libm/etc. are found first,
# preventing RHEL glibc symbols from conflicting with the Ubuntu dynamic linker.
_HOST_LIB = "/host_lib64"
if os.path.isdir(_HOST_LIB) and _HOST_LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    _ldp = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:"
        + (_ldp + ":" if _ldp else "")
        + _HOST_LIB
    )
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import csv
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

# Must be set before habitat_sim is imported so Magnum picks up the NVIDIA EGL driver.
_EGL_LIB = "/host_lib64/libEGL_nvidia.so.0"
if os.path.exists(_EGL_LIB) and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
    import json as _json
    _vendor_json = "/tmp/10_nvidia_egl.json"
    with open(_vendor_json, "w") as _f:
        _json.dump({"file_format_version": "1.0.0",
                    "ICD": {"library_path": _EGL_LIB}}, _f)
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = _vendor_json
else:
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                          "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

try:
    import habitat_sim
    import habitat_sim.utils.common as hab_utils
except ImportError:
    print("ERROR: habitat_sim not found. Run: conda activate habitat")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not found. Run: pip install Pillow")
    sys.exit(1)

try:
    import quaternion as npq
except ImportError:
    print("ERROR: numpy-quaternion not found (should be bundled with habitat-sim)")
    sys.exit(1)

# ─── Paths and constants ───────────────────────────────────────────────────────

DEFAULT_MESH_ROOT    = "/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/meshes"
DEFAULT_GEOMETRY_CSV = (
    Path(__file__).resolve().parents[1] / "matterport3d" / "room_geometry.csv"
)
DEFAULT_OUTPUT_ROOT  = "/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/rendered_images"
DEFAULT_MAPPING_CSV  = (
    Path(__file__).resolve().parents[1] / "matterport3d" / "soundspaces_matterport3d_mapping.csv"
)

IMAGE_H      = 224   # DA2 ViT encoder input height; must be divisible by patch size 14
IMAGE_W      = 294   # DA2 ViT encoder input width;  must be divisible by patch size 14
JPEG_QUALITY = 85
from mp3d_house import (CORNER_OFFSET_M, parse_house_polygons, point_in_polygon,
                        closest_point_on_polygon)

EYE_HEIGHT_FRAC     = 0.85   # fraction of H_m for camera height
TARGET_HEIGHT_FRAC  = 0.50   # fraction of H_m for look-at target height
MAX_CAMERA_HEIGHT_M = 4.0    # ceiling height ceiling for camera placement (eye ≤ 3.4 m)


# ─── Coordinate helpers ────────────────────────────────────────────────────────

def world_to_habitat(world_x: float, world_y: float, world_z: float) -> np.ndarray:
    """
    Convert a Z-up world position (MP3D .house frame) to Habitat-Sim Y-up.

    Habitat loads the .glb through the glTF Z-up→Y-up convention, a −90° rotation
    about X: (x, y, z) → (x, z, −y).  Verified against the shipped .navmesh, whose
    Habitat-frame bounds equal the mesh bounds with the Y range negated.
    """
    return np.array([world_x, world_z, -world_y], dtype=np.float32)


def look_at_quaternion(eye: np.ndarray, target: np.ndarray):
    """
    Full 3D look-at quaternion for Habitat-Sim (Y-up; default camera forward = -Z, right = +X).

    Builds the rotation matrix whose columns are [right, up, -forward] in world space,
    then converts to np.quaternion.
    """
    forward = target - eye
    fwd_norm = np.linalg.norm(forward)
    if fwd_norm < 1e-6:
        return npq.quaternion(1.0, 0.0, 0.0, 0.0)   # identity — eye == target
    forward /= fwd_norm

    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        # Degenerate: camera looks nearly straight up or down — use -Z as fallback up
        world_up = np.array([0.0, 0.0, -1.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Rotation matrix mapping camera default axes to world:
    # Camera +X (right) → world right
    # Camera +Y (up)    → world up
    # Camera +Z         → world -forward  (camera default forward = -Z)
    rot = np.column_stack([right, up, -forward])
    return npq.from_rotation_matrix(rot)


# ─── Camera positions ─────────────────────────────────────────────────────────

def load_rooms_with_valid_rirs(mapping_csv: Path) -> set:
    """
    Return {(scene, region_idx)} for rooms appearing as receiver_region in at least one
    valid mapping row. Rooms with no valid RIR never reach the dataset, so rendering
    overviews for them is wasted work.
    """
    needed = set()
    with open(mapping_csv) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(",")
            if header is None:
                header = {name: i for i, name in enumerate(parts)}
                i_scene  = header["scene"]
                i_valid  = header["valid"]
                i_region = header["receiver_region"]
                continue
            if parts[i_valid] != "True":
                continue
            region = int(parts[i_region])
            if region >= 0:
                needed.add((parts[i_scene], region))
    return needed


def compute_room_cameras(room: dict, polygon: list | None = None) -> list:
    """
    Compute the 4 corner cameras for a room in Habitat Y-up space.
    Returns a list of 4 tuples: (corner_num, eye_hab, target_hab).

    Each camera is placed CORNER_OFFSET_M metres inward from the two nearest corner
    walls. If polygon is provided and the candidate position falls outside it (as in
    L-shaped or T-shaped rooms), the camera is snapped to the nearest point on the
    polygon perimeter then nudged CORNER_OFFSET_M inward toward the polygon centroid.

    room keys: center_x, center_y, L_m, W_m, rotation_deg, z_min, H_m
    """
    cx   = float(room["center_x"])
    cy   = float(room["center_y"])
    L    = float(room["L_m"])
    W    = float(room["W_m"])
    θ    = math.radians(float(room["rotation_deg"]))
    zmin = float(room["z_min"])
    # Stairwells, double-height spaces and open porches report ceiling heights up to
    # ~14 m. Capping the height used for camera geometry keeps the eye at a vantage
    # that still reads as a room view; it is inert for ordinary ceilings.
    H    = min(float(room["H_m"]), MAX_CAMERA_HEIGHT_M)

    eye_world_z = zmin + EYE_HEIGHT_FRAC * H
    tgt_world_z = zmin + TARGET_HEIGHT_FRAC * H
    half_u = max(L / 2 - CORNER_OFFSET_M, 0.0)
    half_v = max(W / 2 - CORNER_OFFSET_M, 0.0)

    use_polygon = polygon is not None and len(polygon) >= 3
    if use_polygon:
        poly_cx = sum(p[0] for p in polygon) / len(polygon)
        poly_cy = sum(p[1] for p in polygon) / len(polygon)

    cameras = []
    # Fixed corner ordering: (+u,+v), (+u,-v), (-u,+v), (-u,-v)
    for num, u_sign, v_sign in [(1, +1, +1), (2, +1, -1), (3, -1, +1), (4, -1, -1)]:
        u, v = u_sign * half_u, v_sign * half_v
        world_x = cx + u * math.cos(θ) - v * math.sin(θ)
        world_y = cy + u * math.sin(θ) + v * math.cos(θ)

        if use_polygon and not point_in_polygon(world_x, world_y, polygon):
            bx, by = closest_point_on_polygon(world_x, world_y, polygon)
            dx, dy = poly_cx - bx, poly_cy - by
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                world_x = bx + dx / dist * CORNER_OFFSET_M
                world_y = by + dy / dist * CORNER_OFFSET_M
            else:
                world_x, world_y = bx, by

        eye_hab    = world_to_habitat(world_x,  world_y, eye_world_z)
        target_hab = world_to_habitat(cx,        cy,      tgt_world_z)
        cameras.append((num, eye_hab, target_hab))

    return cameras


# ─── Simulator ─────────────────────────────────────────────────────────────────

def build_simulator(glb_path: Path) -> habitat_sim.Simulator:
    """Build a Habitat-Sim simulator for a .glb scene with one RGB pinhole sensor."""
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb_path)
    backend.enable_physics = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = [IMAGE_H, IMAGE_W]
    sensor.position = [0.0, 0.0, 0.0]   # default is [0, 1.5, 0]; we specify camera position directly

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def set_agent(sim: habitat_sim.Simulator, position: np.ndarray, rotation):
    """Place the agent at position with rotation."""
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = rotation
    sim.get_agent(0).set_state(state)


# ─── Image I/O ─────────────────────────────────────────────────────────────────

def save_jpeg(arr: np.ndarray, path: Path, quality: int = None):
    """Save RGBA or RGB uint8 array as JPEG."""
    img = arr[:, :, :3] if arr.shape[2] == 4 else arr
    q = JPEG_QUALITY if quality is None else quality
    Image.fromarray(img.astype(np.uint8)).save(path, format="JPEG", quality=q)


# ─── Per-scene processing ──────────────────────────────────────────────────────

def render_room(sim: habitat_sim.Simulator, room: dict,
                polygon: list | None,
                out_dir: Path, skip_existing: bool) -> tuple[int, int]:
    """
    Render 4 corner cameras for one room. Returns (n_rendered, n_skipped).
    polygon: floor polygon vertices [(x,y), ...] for point-in-polygon fallback, or None.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = compute_room_cameras(room, polygon)
    rendered = skipped = 0

    for corner_num, eye_hab, target_hab in cameras:
        out_path = out_dir / f"corner{corner_num}_rgb.jpg"
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        rot = look_at_quaternion(eye_hab, target_hab)
        set_agent(sim, eye_hab, rot)
        obs = sim.get_sensor_observations()
        save_jpeg(obs["rgb"], out_path)
        rendered += 1

    return rendered, skipped


def process_scene(scene: str, rooms: list, glb_path: Path,
                  output_root: Path, skip_existing: bool) -> tuple[int, int]:
    """
    Render overview images for all rooms in one scene.
    Opens the simulator once, then renders each room. Returns (n_rendered, n_skipped).
    """
    house_path = glb_path.parent / f"{scene}.house"
    polygons = parse_house_polygons(house_path)

    sim = build_simulator(glb_path)
    total_rendered = total_skipped = 0

    for room in rooms:
        ridx    = int(room["region_idx"])
        out_dir = output_root / "room_overviews" / f"{scene}_r{ridx:03d}"
        rendered, skipped = render_room(sim, room, polygons.get(ridx), out_dir, skip_existing)
        total_rendered += rendered
        total_skipped  += skipped

    sim.close()
    return total_rendered, total_skipped


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render MP3D per-room corner overview images via Habitat-Sim."
    )
    parser.add_argument("--mesh-root",    type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument("--output-root",  type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenes",       type=str,  default=None,
                        help="Comma-separated scene names (default: all in geometry CSV)")
    parser.add_argument("--no-skip",      action="store_true",
                        help="Re-render even if output already exists")
    parser.add_argument("--mapping-csv",  type=Path, default=DEFAULT_MAPPING_CSV,
                        help="Mapping CSV used to skip rooms that have no valid RIRs")
    parser.add_argument("--all-rooms",    action="store_true",
                        help="Render every room in the geometry CSV, including rooms with no valid RIRs")
    parser.add_argument("--image-size", type=str, default=None,
                        help="Override render size as H,W — both divisible by the ViT patch size 14")
    parser.add_argument("--jpeg-quality", type=int, default=None,
                        help="Override JPEG quality")
    args = parser.parse_args()

    global IMAGE_H, IMAGE_W, JPEG_QUALITY
    if args.image_size:
        IMAGE_H, IMAGE_W = (int(v) for v in args.image_size.split(","))
    if args.jpeg_quality:
        JPEG_QUALITY = args.jpeg_quality

    # Load geometry CSV, group rooms by scene
    print("Reading geometry CSV...", end=" ", flush=True)
    rooms_by_scene: dict[str, list] = {}
    with open(args.geometry_csv, newline="") as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            sc = row["scene"]
            if sc not in rooms_by_scene:
                rooms_by_scene[sc] = []
            rooms_by_scene[sc].append(row)

    total_rooms = sum(len(v) for v in rooms_by_scene.values())
    print(f"done  ({total_rooms} rooms across {len(rooms_by_scene)} scenes)")

    if not args.all_rooms and args.mapping_csv.exists():
        print("Filtering to rooms with valid RIRs...", end=" ", flush=True)
        needed = load_rooms_with_valid_rirs(args.mapping_csv)
        rooms_by_scene = {
            sc: kept
            for sc, rr in rooms_by_scene.items()
            if (kept := [r for r in rr if (sc, int(r["region_idx"])) in needed])
        }
        kept_rooms = sum(len(v) for v in rooms_by_scene.values())
        print(f"done  ({kept_rooms} kept, {total_rooms - kept_rooms} skipped)")

    scenes = ([s.strip() for s in args.scenes.split(",")]
              if args.scenes else sorted(rooms_by_scene))

    print(f"\nRendering room overviews  |  {len(scenes)} scene(s)  |  {IMAGE_H}×{IMAGE_W} JPEG")
    print(f"Output root: {args.output_root}\n")

    args.output_root.mkdir(parents=True, exist_ok=True)

    errors = []
    total_rendered = total_skipped = 0
    t_start = time.time()

    for i, scene in enumerate(scenes, 1):
        glb_path = args.mesh_root / scene / f"{scene}.glb"
        if not glb_path.exists():
            msg = f".glb not found: {glb_path}"
            print(f"  [{i}/{len(scenes)}] {scene}: ERROR — {msg}")
            errors.append((scene, msg))
            continue

        rooms = rooms_by_scene.get(scene, [])
        if not rooms:
            print(f"  [{i}/{len(scenes)}] {scene}: no rooms in geometry CSV, skipping")
            continue

        try:
            rendered, skipped = process_scene(scene, rooms, glb_path,
                                              args.output_root, not args.no_skip)
        except Exception as e:
            print(f"  [{i}/{len(scenes)}] {scene}: ERROR — {e}")
            errors.append((scene, str(e)))
            continue

        total_rendered += rendered
        total_skipped  += skipped

        elapsed   = time.time() - t_start
        remaining = (elapsed / i) * (len(scenes) - i)
        print(f"  [{i}/{len(scenes)}] {scene}: {rendered} rendered, {len(rooms)} rooms"
              + (f", {skipped} skipped" if skipped else "")
              + f"  |  elapsed {timedelta(seconds=int(elapsed))}"
              + f"  |  ETA {timedelta(seconds=int(remaining))}")

    suffix = "_incomplete" if errors else ""
    summary_path = args.output_root / f"overviews_render_summary{suffix}.txt"
    lines = [
        "MP3D room_overviews render summary",
        "=" * 40,
        f"Scenes processed: {len(scenes) - len(errors)} / {len(scenes)}",
        f"Total rendered:   {total_rendered:,}",
        f"Total skipped:    {total_skipped:,}",
        f"Resolution:       {IMAGE_H}×{IMAGE_W} JPEG q{JPEG_QUALITY}",
        f"Cameras per room: 4  (CORNER_OFFSET_M={CORNER_OFFSET_M}m, eye={EYE_HEIGHT_FRAC}·H, "
        f"target={TARGET_HEIGHT_FRAC}·H, H capped at {MAX_CAMERA_HEIGHT_M}m)",
    ]
    if errors:
        lines += ["", f"FAILED SCENES ({len(errors)}):"]
        for s, e in errors:
            lines.append(f"  {s}: {e}")
    summary_path.write_text("\n".join(lines) + "\n")

    if errors:
        print(f"\n{'!'*60}")
        print(f"  WARNING: {len(errors)} scene(s) FAILED — summary: {summary_path}")
        for s, e in errors:
            print(f"    {s}: {e}")
        print(f"{'!'*60}")
    else:
        print(f"\nDone. {total_rendered:,} rendered, {total_skipped:,} skipped.")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
