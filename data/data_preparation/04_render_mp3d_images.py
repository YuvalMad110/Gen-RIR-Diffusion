"""
04_render_mp3d_images.py
-------------------------
Renders per-pair RGB images for SoundSpaces/Matterport3D via Habitat-Sim.

Camera placed at the receiver position, facing the source — one 224×294 JPEG
per (receiver, source) pair where valid=True AND same_region=True.

Coordinate system note
----------------------
MP3D points.txt stores navigable positions at z ≈ 1.52 m (ear/standing height
above the floor). Unlike Replica (z ≈ 0.009 m), NO EYE_HEIGHT offset is added.
Habitat-Sim uses Y-up; the transform is simply:
    habitat_pos = (pts_x, pts_z, pts_y)   # swap Y and Z

Output
------
  {output_root}/target2source_rgb/{scene}/tgt{r}_src{s}.jpg

Usage
-----
    conda activate habitat

    # smoke test — 10 pairs from one scene
    python3 data/data_preparation/04_render_mp3d_images.py \\
        --scenes 17DRP5sb8fy --max-pairs 10

    # full render
    python3 data/data_preparation/04_render_mp3d_images.py
"""

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

# ─── Paths and constants ───────────────────────────────────────────────────────

DEFAULT_MESH_ROOT    = "/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/meshes"
DEFAULT_MAPPING_CSV  = (
    Path(__file__).resolve().parents[1] / "matterport3d"
    / "soundspaces_matterport3d_mapping.csv"
)
DEFAULT_OUTPUT_ROOT  = "/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/rendered_images"

IMAGE_H      = 224   # DA2 ViT encoder input height; must be divisible by patch size 14
IMAGE_W      = 294   # DA2 ViT encoder input width;  must be divisible by patch size 14
JPEG_QUALITY = 85


# ─── Coordinate helpers ────────────────────────────────────────────────────────

def pts_to_habitat(pos: tuple) -> np.ndarray:
    """
    Convert a position from points.txt (Z-up) to Habitat-Sim (Y-up).

    Habitat loads the .glb through the glTF Z-up→Y-up convention, a −90° rotation
    about X: (x, y, z) → (x, z, −y).  Verified against the shipped .navmesh, whose
    Habitat-frame bounds equal the mesh bounds with the Y range negated.

    MP3D navigable points sit at z ≈ 1.52 m (ear height) — no offset added.
    """
    return np.array([pos[0], pos[2], -pos[1]], dtype=np.float32)


def facing_quaternion(receiver_pos: tuple, source_pos: tuple):
    """
    Quaternion for a camera at receiver_pos facing source_pos in the horizontal plane.
    Height difference is ignored; camera is levelled (no pitch).
    """
    r = pts_to_habitat(receiver_pos)
    s = pts_to_habitat(source_pos)
    direction = s - r
    direction[1] = 0   # project to XZ horizontal plane (Y-up)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return hab_utils.quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
    direction /= norm
    # Habitat default forward = -Z; after yaw θ around Y: forward = (-sinθ, 0, -cosθ).
    # Solve: direction = (-sinθ, 0, -cosθ) → θ = atan2(-dx, -dz)
    angle = math.atan2(-float(direction[0]), -float(direction[2]))
    return hab_utils.quat_from_angle_axis(angle, np.array([0.0, 1.0, 0.0]))


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
    sensor.resolution = [IMAGE_H, IMAGE_W]   # [rows, cols] = [H, W]
    sensor.position = [0.0, 0.0, 0.0]   # default is [0, 1.5, 0]; MP3D points are at eye height already

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
    """Save RGBA or RGB uint8 array as JPEG at the given quality."""
    img = arr[:, :, :3] if arr.shape[2] == 4 else arr
    q = JPEG_QUALITY if quality is None else quality
    Image.fromarray(img.astype(np.uint8)).save(path, format="JPEG", quality=q)


# ─── Per-scene processing ──────────────────────────────────────────────────────

def process_scene(scene: str, glb_path: Path, pairs: list,
                  out_dir: Path, skip_existing: bool) -> tuple[int, int]:
    """
    Render all target2source pairs for one scene.
    Returns (n_rendered, n_skipped).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    skipped = 0
    for pair in pairs:
        out_path = out_dir / f"tgt{pair['receiver_idx']}_src{pair['source_idx']}.jpg"
        if skip_existing and out_path.exists():
            skipped += 1
        else:
            todo.append((pair, out_path))

    if not todo:
        return 0, skipped

    sim = build_simulator(glb_path)
    rendered = 0

    for pair, out_path in todo:
        hab_pos = pts_to_habitat(pair["receiver_pos"])
        hab_rot = facing_quaternion(pair["receiver_pos"], pair["source_pos"])
        set_agent(sim, hab_pos, hab_rot)
        obs = sim.get_sensor_observations()
        save_jpeg(obs["rgb"], out_path)
        rendered += 1

    sim.close()
    return rendered, skipped


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render MP3D per-pair target2source RGB images via Habitat-Sim."
    )
    parser.add_argument("--mesh-root",   type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenes",      type=str,  default=None,
                        help="Comma-separated scene names (default: all in mapping CSV)")
    parser.add_argument("--max-pairs",   type=int,  default=None,
                        help="Max pairs per scene — for smoke tests")
    parser.add_argument("--no-skip",     action="store_true",
                        help="Re-render even if output already exists")
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

    # Read CSV once, filter to valid+same_region, group by scene.
    # The mapping CSV is 1.7 GB — reading it once avoids O(scenes²) file I/O.
    print("Reading mapping CSV...", end=" ", flush=True)
    pairs_by_scene: dict[str, list] = {}
    with open(args.mapping_csv, newline="") as f:
        reader = csv.DictReader(r for r in f if not r.startswith("#"))
        for row in reader:
            if row["valid"] != "True" or row["same_region"] != "True":
                continue
            sc = row["scene"]
            if sc not in pairs_by_scene:
                pairs_by_scene[sc] = []
            if args.max_pairs and len(pairs_by_scene[sc]) >= args.max_pairs:
                continue
            pairs_by_scene[sc].append({
                "receiver_idx": int(row["receiver_idx"]),
                "source_idx":   int(row["source_idx"]),
                "receiver_pos": (float(row["receiver_x"]),
                                 float(row["receiver_y"]),
                                 float(row["receiver_z"])),
                "source_pos":   (float(row["source_x"]),
                                 float(row["source_y"]),
                                 float(row["source_z"])),
            })
    total_pairs = sum(len(v) for v in pairs_by_scene.values())
    print(f"done  ({total_pairs:,} valid+same_region pairs across {len(pairs_by_scene)} scenes)")

    scenes = ([s.strip() for s in args.scenes.split(",")]
              if args.scenes else sorted(pairs_by_scene))

    print(f"\nRendering {len(scenes)} scene(s)  |  {IMAGE_H}×{IMAGE_W} JPEG  |  "
          f"filter: valid+same_region")
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

        pairs = pairs_by_scene.get(scene, [])
        if not pairs:
            print(f"  [{i}/{len(scenes)}] {scene}: no valid+same_region pairs, skipping")
            continue

        out_dir = args.output_root / "target2source_rgb" / scene

        try:
            rendered, skipped = process_scene(scene, glb_path, pairs, out_dir,
                                              not args.no_skip)
        except Exception as e:
            print(f"  [{i}/{len(scenes)}] {scene}: ERROR — {e}")
            errors.append((scene, str(e)))
            continue

        total_rendered += rendered
        total_skipped  += skipped

        elapsed   = time.time() - t_start
        remaining = (elapsed / i) * (len(scenes) - i)
        print(f"  [{i}/{len(scenes)}] {scene}: {rendered} rendered"
              + (f", {skipped} skipped" if skipped else "")
              + f"  |  elapsed {timedelta(seconds=int(elapsed))}"
              + f"  |  ETA {timedelta(seconds=int(remaining))}")

    # Write summary (filename gets _incomplete suffix if any scene failed)
    suffix = "_incomplete" if errors else ""
    summary_path = args.output_root / f"target2source_render_summary{suffix}.txt"
    lines = [
        "MP3D target2source_rgb render summary",
        "=" * 40,
        f"Scenes processed: {len(scenes) - len(errors)} / {len(scenes)}",
        f"Total rendered:   {total_rendered:,}",
        f"Total skipped:    {total_skipped:,}",
        f"Resolution:       {IMAGE_H}×{IMAGE_W} JPEG q{JPEG_QUALITY}",
        f"Filter:           valid=True AND same_region=True",
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
