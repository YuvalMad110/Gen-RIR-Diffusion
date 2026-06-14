# Gen-RIR-Diffusion: Image Conditioning Project

## Goal
Add visual room image conditioning to the diffusion model using SoundSpaces RIRs matched to Replica 3D scene renderings.

## Step Progress

- [x] **Step 0** — CLAUDE.md created
- [x] **Step 1** — Download Replica + SoundSpaces data (see `DATASET_PREPARATION.md`)
- [x] **Step 2** — Explore SoundSpaces positions (`data/data_preparation/01_explore_soundspaces.py`)
- [x] **Step 3** — Preprocess SoundSpaces mapping (`data/data_preparation/02_preprocess_soundspaces.py` → `data/soundspaces_replica_mapping.csv`)
- [x] **Step 4** — Build ReplicaPoseRenderer C++ binary (`tools/setup_replica_renderer.sh`)
- [x] **Step 5** — Render per-pair RGB images for all 18 scenes (`data/data_preparation/04_render_replica_poses.py` → `Replica_rendered/target2source_rgb/`)
- [x] **Step 6** — Compute room geometry for all 18 scenes (`data/data_preparation/06_compute_room_geometry.py` → `data/room_geometry.csv`)
- [x] **Step 7** — Room overview images (`data/data_preparation/07_render_room_overviews.py` + `08_render_mid_overviews.py` → 12 views per scene: `top/mid/bottom_corner{1-4}_rgb/depth`)
- [ ] **Step 8** — Per-pair depth render in progress → `/dsi/fetaya-lab/dataset/Replica_rendered/target2source_depth/`
- [x] **Step 9** — SoundSpaces dataset class (`data/soundspaces_replica.py`) + collate (`data/dataset_collate_fn.py`) + split util (`utils/dataset_utils.py`)
- [x] **Step 10** — Image encoder: DA3 ViT-L backbone, pyramid features, fuse/stack modes (`image_encoder.py`); design doc at `docs/image_conditioning_design.md`
- [x] **Step 11** — Image conditioning in model: `FusionBlock`, `image_encoder` param, updated `forward`/`generate`/`get_null_conditioning` (`RIRDiffusionModel.py`); trainer dict-batch unpacking (`trainer.py`); `run_train.py` SoundSpaces support + `model_config_VisualCond.json`
- [x] **Step 12** — Multi-image fetching: `rir_view_type` + `room_overview_types` in `data/soundspaces_replica.py`; depth loading is placeholder (repeat 3×) — see Step 15
- [ ] **Step 13** — First training run (scalar-only SoundSpaces baseline, then with images)
- [ ] **Step 14** — Encoding strategy: flat vs. separate encoders for pair-view vs. scene-overview images; `out_dim` ablation (128 → 256/512)
- [ ] **Step 15** — Proper depth input handling in `ImageEncoder`: learned `Conv2d(1, 3, kernel_size=1)` input projection before backbone (Option A); current repeat-3× in `_load_depth` is a placeholder
- [x] **Step 16** — Add `--use_rt60_condition` flag for SoundSpaces training: estimate RT60 from each RIR at dataset load time and append it to the 9-dim condition vector, making it 10-dim and matching GTU's format.

## Key Paths
- SoundSpaces metadata: `/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/metadata/replica/`
- SoundSpaces RIRs: `/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica/`
- Replica scenes: `/dsi/gannot-lab/gannot-lab1/datasets/Replica/`
- Rendered images (production): `/dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered/`
- Rendered images (test): `~/temp/Replica_rendered/`
- Project: `/home/yuvalmad/Projects/Gen-RIR-Diffusion/`

## Coordinate System

### Shared axes: SoundSpaces points.txt and Replica mesh
- Both use the same X, Y, Z axes — no conversion between datasets.
- X, Y = floor plane; Z = up (floor at z_min, ceiling at z_max).
- pts_z is the camera/ear height in the scene (roughly 1.5 m above the floor surface).

### Pangolin renderer up vector
Pass `up=(0, 0, 1)` to `ModelViewLookAtRDF`.

### Room-aligned coordinates (for model conditioning)
The room walls are NOT axis-aligned with mesh X/Y. `data/room_geometry.csv` provides per-scene
rotation angle θ, center (cx, cy), and dimensions L, W, H. To express positions in room-aligned
metres (u along long axis, v along short axis):
```python
dx, dy = pts_x - center_x, pts_y - center_y
u =  dx*cos(θ) + dy*sin(θ)   # metres along long axis
v = -dx*sin(θ) + dy*cos(θ)   # metres along short axis
```

## Room Geometry (`data/room_geometry.csv`)
Computed by `data/data_preparation/06_compute_room_geometry.py` from mesh.ply files.

**Algorithm:**
1. Select vertices within 0.1 m of z_min → floor surface
2. Compute 2D convex hull of floor footprint (scipy.spatial.ConvexHull)
3. Find minimum-area bounding rectangle via rotating calipers → L (long), W (short), θ (rotation)
4. Room height H = z_max − z_min

**Columns:** `scene, L_m, W_m, H_m, center_x, center_y, rotation_deg, hull_area_m2, rect_area_m2`

Verification PNGs (hull + bounding rect overlay): `data/replica_room_geometry/{scene}.png`

## Dataset Structure Notes

### SoundSpaces
- RIRs at: `[scene]/[angle_deg]/[receiver_idx]_[source_idx].wav`
- Positions: `[scene]/points.txt` — tab-separated `index x y z` (Z-up coords)
- Graph: `[scene]/graph.pkl` — NetworkX graph; nodes = positions with RIRs; edges = nav adjacency
- RIRs exist between ALL graph-node pairs (not just adjacent edges)
- 18 scenes, 1732 graph nodes total, ~225K directed pairs per angle, 4 angles (0/90/180/270)
- Mono-RIR approach: `soundspaces_replica_mapping.csv` selects best_angle + best_channel per pair

### Replica
- 18 indoor scenes: apartments, offices, hotel, rooms
- Each scene: `mesh.ply`, `textures/`, `glass.sur` (mirror surfaces)
- Custom renderer: `tools/ReplicaSDK/src/replica_pose_renderer.cpp` (PTex HDR, 1280×960)
- Build: `tools/setup_replica_renderer.sh` → binary at `~/tools/Replica-Dataset/build/ReplicaSDK/ReplicaPoseRenderer`

### Rendered Images
- `target2source_rgb/{scene}/tgt{r}_src{s}.jpg` — camera at receiver, looking toward source
- `target2source_depth/{scene}/tgt{r}_src{s}.png` — 16-bit depth

## Model Conditioning Plan
- Current: 10-dim vector [room_dims(3), mic_loc(3), speaker_loc(3), rt60(1)]
- Planned: room_dims = (L, W, H) from room_geometry.csv; positions in room-aligned (u, v) coords
- Image conditioning: TBD (Step 10 research — candidates: CLIP ViT-B/32, DINOv2, Places365-ResNet)
- Fusion: concatenate projected image features to existing cond vector

## Python Version
Always use `python3` and `python3 -m pip install` (Python 3.12).
