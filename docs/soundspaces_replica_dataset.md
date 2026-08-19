# SoundSpaces + Replica Dataset Reference

## Key Paths

- SoundSpaces metadata: `/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/metadata/replica/`
- SoundSpaces RIRs: `/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica/`
- Replica scenes: `/dsi/gannot-lab/gannot-lab1/datasets/Replica/`
- Rendered images (production): `/dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered/`
- Rendered images (test): `~/temp/Replica_rendered/`

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

## Dataset Structure

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
