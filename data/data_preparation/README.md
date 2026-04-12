# Data Preparation Scripts

Scripts for validating, preprocessing, and rendering the SoundSpaces + Replica datasets.
Run these in order when setting up the dataset pipeline from scratch.

| Script | Step | Description |
|--------|------|-------------|
| `01_explore_soundspaces.py` | Validation | Validates all 18 SoundSpaces scenes, counts positions/graph nodes/RIRs, saves summary to `data/data_analysis/soundspaces_replica_summary.txt` |
| `02_preprocess_soundspaces.py` | Preprocessing | For every (receiver, source) pair: computes bearing, selects best heading angle and ear channel, writes `data/soundspaces_replica_mapping.csv` |
| `03_render_replica_images.py` | *(superseded)* | Earlier Habitat-Sim-based renderer. Kept for reference; not part of the active pipeline. |
| `04_render_replica_poses.py` | Rendering | Renders per-pair RGB/depth images using ReplicaPoseRenderer (PTex, 1280×960). Camera at receiver position looking toward source. Output: `target2source_rgb/{scene}/tgt{r}_src{s}.jpg` |
| `06_compute_room_geometry.py` | Room geometry | Computes per-scene bounding rectangle (L, W, H, center, rotation, z_min) from mesh.ply. Output: `data/room_geometry.csv` + verification PNGs in `data/replica_room_geometry/` |
| `07_render_room_overviews.py` | Overview rendering | Renders 8 overview images per scene (4 XY corners × above ceiling / below floor). Output: `room_overviews/{scene}/top_corner{1-4}_rgb.jpg`, `bottom_corner{1-4}_rgb.jpg` |

See `DATASET_PREPARATION.md` at the project root for full setup instructions (downloads, installations, environment).

## Quick start

```bash
# Step 1 — validate data
python3 data/data_preparation/01_explore_soundspaces.py

# Step 2 — generate mapping CSV
python3 data/data_preparation/02_preprocess_soundspaces.py

# Step 3 — compute room geometry for all 18 scenes
python3 data/data_preparation/06_compute_room_geometry.py

# Step 4 — test render (office_2, 10 pairs, RGB only)
python3 data/data_preparation/04_render_replica_poses.py \
    --scenes office_2 --max-pairs 10 --types rgb

# Step 5 — full production render (all scenes, RGB + depth)
python3 data/data_preparation/04_render_replica_poses.py --types rgb,depth
python3 data/data_preparation/07_render_room_overviews.py --types rgb,depth
```

## Requirements

- `ReplicaPoseRenderer` binary built at `~/tools/Replica-Dataset/build/ReplicaSDK/ReplicaPoseRenderer`
- Run `tools/setup_replica_renderer.sh` to build it
- NVIDIA GPU with EGL support
