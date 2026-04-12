# Dataset Preparation Guide

This document describes how to reproduce the full dataset pipeline from scratch for the Gen-RIR-Diffusion image-conditioning project. The pipeline requires two datasets: **SoundSpaces** (binaural RIRs + position metadata) and **Replica** (3D scene meshes for image rendering).

---

## Overview

| Dataset | Purpose | Size |
|---------|---------|------|
| SoundSpaces (Replica split) | Pre-computed binaural RIRs at discrete positions in 18 indoor scenes | ~large |
| Replica | 3D mesh scenes used to render RGB/depth images at SoundSpaces positions | ~30 GB |

Pipeline:
1. Download SoundSpaces binaural RIRs + metadata
2. Download Replica 3D meshes
3. Build ReplicaPoseRenderer binary
4. Generate mapping CSV (preprocessing)
5. Compute room geometry
6. Render per-pair images and room overview images

---

## Step 1: SoundSpaces — Binaural RIRs + Metadata

SoundSpaces provides pre-computed RIRs inside 3D scenes. We use the **Replica** scene split with **binaural** RIRs.

### Clone and download

```bash
git clone https://github.com/facebookresearch/sound-spaces.git
```

Follow the data download instructions in the SoundSpaces repo. You need two components:

**a) Binaural RIRs** — organized by scene / heading angle / receiver-source pair:
```
binaural_rirs/replica/
  apartment_0/
    0/          # heading angle 0°
      101_10.wav
      ...
    90/  180/  270/
  ... (18 scenes total)
```
File naming: `[receiver_idx]_[source_idx].wav`. Format: binaural stereo, 44.1 kHz.

**b) Metadata** — position and connectivity data per scene:
```
metadata/replica/
  apartment_0/
    points.txt    # navigable positions: tab-separated  index  x  y  z
    graph.pkl     # NetworkX graph — nodes = positions with RIRs
  ... (18 scenes total)
```

### Dataset statistics
- 18 scenes: `apartment_0-2`, `frl_apartment_0-5`, `hotel_0`, `office_0-4`, `room_0-2`
- ~3,400 navigable positions total
- 4 heading angles per position: 0°, 90°, 180°, 270°
- RIRs exist between all graph-node pairs

See `data/data_analysis/soundspaces_replica_summary.txt` for the full per-scene breakdown.

---

## Step 2: Replica — 3D Scene Meshes

### Clone and download

```bash
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
bash download.sh /path/to/datasets/Replica
```

Downloads ~30 GB. If interrupted, re-run — `wget --continue` resumes incomplete parts.

Each scene contains:
```
Replica/
  apartment_0/
    mesh.ply       # full scene mesh with per-vertex colour
    textures/      # PTex texture atlases
    glass.sur      # glass/mirror surface definitions
  ... (18 scenes total)
```

---

## Step 3: Build ReplicaPoseRenderer

The renderer is a custom C++ binary built on top of the Replica SDK.

```bash
bash tools/setup_replica_renderer.sh
```

Binary output: `~/tools/Replica-Dataset/build/ReplicaSDK/ReplicaPoseRenderer`

Requires: CUDA, EGL, CMake. See `tools/setup_replica_renderer.sh` for dependencies.

---

## Step 4: Preprocessing — Mapping CSV

Generates `data/soundspaces_replica_mapping.csv` — one row per (receiver, source) pair with
the best heading angle and ear channel selected from the four available SoundSpaces angles.

```bash
python3 data/data_preparation/02_preprocess_soundspaces.py
```

---

## Step 5: Room Geometry

Computes per-scene bounding rectangle dimensions, center, rotation angle, and floor Z from
the mesh.ply files. Output: `data/room_geometry.csv`.

```bash
python3 data/data_preparation/06_compute_room_geometry.py
```

---

## Step 6: Render Images

### Per-pair images (camera at receiver, facing source)

```bash
# Test (office_2, 10 pairs, RGB only)
python3 data/data_preparation/04_render_replica_poses.py \
    --scenes office_2 --max-pairs 10 --types rgb

# Full production render (~5 hrs RGB, ~10 hrs RGB+depth)
python3 data/data_preparation/04_render_replica_poses.py --types rgb,depth
```

Output:
```
Replica_rendered/
  target2source_rgb/{scene}/tgt{r}_src{s}.jpg      # 1280×960 JPEG
  target2source_depth/{scene}/tgt{r}_src{s}.png    # 16-bit depth PNG
```

### Room overview images (8 corners per scene)

```bash
python3 data/data_preparation/07_render_room_overviews.py --types rgb,depth
```

Output:
```
Replica_rendered/
  room_overviews/{scene}/top_corner{1-4}_rgb.jpg
  room_overviews/{scene}/bottom_corner{1-4}_rgb.jpg
  (and _depth.png equivalents)
```

---

## Canonical Data Paths

All scripts use these default paths (override via CLI arguments):

| Resource | Default path |
|----------|-------------|
| Replica scenes | `/dsi/gannot-lab/gannot-lab1/datasets/Replica/` |
| SoundSpaces metadata | `/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/metadata/replica/` |
| SoundSpaces RIRs | `/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica/` |
| Rendered images | `/dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered/` |

---

## Preparation Scripts

All scripts live in `data/data_preparation/`. See that folder's `README.md` for a full index.
