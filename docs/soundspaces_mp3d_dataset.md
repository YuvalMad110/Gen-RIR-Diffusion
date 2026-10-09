# SoundSpaces + Matterport3D Dataset Reference

## Key Paths

- SoundSpaces metadata:   `/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/metadata/mp3d/`
- SoundSpaces RIRs:       `/dsi/fetaya-lab/yuvalmad/datasets/SoundSpaces/binaural_rirs/mp3d/`
- Scene meshes:           `/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/meshes/`
- Rendered images:        `/dsi/fetaya-lab/yuvalmad/datasets/Matterport3D/rendered_images/`
- Raw download (zip):     `/home/yuvalmad/datasets/Matterport3D/v1/tasks/mp3d_habitat.zip`
- Mapping CSV:            `data/matterport3d/soundspaces_matterport3d_mapping.csv`
- Room geometry CSV:      `data/matterport3d/room_geometry.csv`

## Dataset Scale

- 85 scenes in SoundSpaces (90 in the full Matterport3D dataset)
- 4 azimuths per pair: 0 / 90 / 180 / 270 degrees
- Navigable points per scene: varies widely (~200 to ~1600; see points.txt)
- RIR pairs per scene: subset of all-pairs, determined by `graph.pkl` connectivity

## Coordinate System

### SoundSpaces points.txt
Same format as Replica: tab-separated `index  x  y  z`.

Key difference from Replica: navigable points sit at **z ≈ 1.52 m** (standing/ear height above
the floor), whereas Replica points are at z ≈ 0.009 m (floor-level sensor). The mesh coordinate
origin is near the floor (z_min ≈ 0), so the `_to_room_coords` formula is unchanged:

```python
dx, dy = pts_x - center_x, pts_y - center_y
u =  dx*cos(θ) + dy*sin(θ)   # metres along long axis
v = -dx*sin(θ) + dy*cos(θ)   # metres along short axis
z_norm = pts_z - z_min        # ≈ 1.52 m for all MP3D navigable points
```

### Habitat-Sim rendering coordinate transform
Habitat-Sim uses Y-up; points.txt uses Z-up. Transform applied in render script:
```python
habitat_pos = (pts_x, pts_z, pts_y)   # (X, Z_pts→Y_hab, Y_pts→Z_hab)
```
Same convention as the Replica render script (`03_render_replica_images.py`).

## Room Geometry (`data/matterport3d/room_geometry.csv`)

Computed by `data/data_preparation/13_compute_mp3d_geometry.py` from the **`.house` file**
(not from the PLY mesh). The `.house` file provides pre-computed, authoritative geometry
for each room, avoiding the vertex z_min heuristic used for Replica.

**Algorithm:**
1. Parse `V` (vertex) lines: room boundary polygon vertices at their true floor height, grouped by room
2. Union polygon vertices across all rooms on the primary level → scene-level floor footprint
3. Compute minimum-area bounding rectangle via rotating calipers → L (long), W (short), θ (rotation)
4. Floor height z_min from `L` (level) line `bbox_min_z`; ceiling z_max from `bbox_max_z`; H = z_max − z_min

**Columns:** `scene, L_m, W_m, H_m, center_x, center_y, rotation_deg, hull_area_m2, rect_area_m2`
(same schema as `data/replica/room_geometry.csv`)

Why `.house` instead of PLY:
- Floor/ceiling heights are explicit per room (MP3D is multi-floor; z_min of all vertices would pick the wrong floor)
- Room boundary polygons are pre-computed — more accurate than convex hull of 1.5M mesh vertices
- Much faster (text parsing vs. binary PLY iteration)

## Dataset Structure

### SoundSpaces RIRs
- Path: `{scene}/data/binaural_rirs/mp3d/{scene}/{azimuth_deg}/{receiver_idx}_{source_idx}.wav`
- Same binaural `.wav` format as Replica (2-channel, 22050 Hz)

### SoundSpaces metadata
- Positions: `{scene}/points.txt` — tab-separated `index x y z`
- Graph: `{scene}/graph.pkl` — NetworkX graph; nodes = positions with RIRs; edges = nav adjacency
- RIRs exist between ALL graph-node pairs (same as Replica)

### Scene meshes (from `mp3d_habitat.zip`)
Per scene under `meshes/{scene}/`:
- `{scene}.glb` — full textured mesh for Habitat-Sim rendering
- `{scene}_semantic.ply` — same vertex geometry as `.glb`, with semantic object labels; used for geometry verification if needed
- `{scene}.house` — structured room geometry: levels, regions, floor polygons, vertex lists
- `{scene}.navmesh` — Habitat-Sim navigation mesh

Note: `house_segmentations` PLY (from the separate Matterport3D `scans/` download) is **not
needed**. The `_semantic.ply` has identical vertex count and XYZ coordinates (verified on
`17DRP5sb8fy`: 1,522,546 vertices in both); only per-vertex attributes differ.

### Rendered images
Under `rendered_images/` — populated by `04_render_mp3d_images.py` and `05_render_mp3d_overviews.py`.

**Two image types:**

**Per-pair (target2source):** one image per valid+same_region pair — camera at receiver
position facing source, rendered via Habitat-Sim with the `.glb` mesh.
- `target2source_rgb/{scene}/tgt{r}_src{s}.jpg` — 392×518 JPEG

**Per-room overviews:** 4 corner cameras per room, placed at room-aligned positions derived
from `room_geometry.csv` (center, L, W, rotation_deg, z_min, H_m). Each camera is inset to
65% of L/2 and W/2 from the room center (to stay inside irregular room footprints) and placed
at 85% of ceiling height, looking toward the room center.
- `room_overviews/{scene}_r{ridx:03d}/corner{1-4}_rgb.jpg` — 392×518 JPEG

**Image resolution: 392×518 (H×W)**

The DA2 ViT backbone (patch size 14) requires both H and W divisible by 14.
392 = 28×14, 518 = 37×14. Aspect ratio ≈ 4:3 (W:H landscape), matching Habitat-Sim's
standard output and the Replica renders (1280×960 stored, resized to 392×518 at load time).

For MP3D we store directly at encoder input resolution (392×518) to save storage.
Alternatives that were considered and rejected:

| Resolution | Reason rejected |
|---|---|
| 1280×960 (Replica stored size) | ~770 GB for 966K pairs — exceeds storage budget |
| 64×64 (AV-RIR) | Below minimum useful resolution for ViT; would destroy features |
| 224×308 | Lower quality than model was trained with; no compelling storage benefit |
| **392×518** | Matches model's native input; ~116 GB for pairs + ~1 GB overviews — fits budget |

Uniform resolution is used for all image types (per-pair and overview) so the image encoder
sees identically-sized inputs regardless of image source.

**Coordinate transform for MP3D rendering**

MP3D points.txt already stores navigable points at z ≈ 1.52 m (ear/standing height).
Unlike Replica (where z ≈ 0.009 m requires adding EYE_HEIGHT), no height offset is applied:
```python
habitat_pos = np.array([pts_x, pts_z, pts_y])  # Z-up → Y-up; no EYE_HEIGHT added
```
For overview cameras, world positions (computed in Z-up from room_geometry.csv) are converted
the same way: `habitat = (world_x, world_z, world_y)`.

**Alignment between images and scalar conditioning**

Overview cameras are defined in room-aligned coordinates (derived from center, L, W,
rotation_deg), the same coordinate system as mic_loc/speaker_loc scalars. Camera 1 is always
at the (+u, +v) corner in room space. A receiver at room coord (+u, +v) therefore always
appears in the foreground of camera 1 — the model can learn this spatial relationship without
any implicit geometry reasoning. Receiver and source positions are additionally projected as
coloured dots onto each overview image at dataset load time (green = receiver, red = source)
to make the spatial relationship explicit without architectural changes.
