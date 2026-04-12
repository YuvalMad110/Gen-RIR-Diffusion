"""
01_explore_soundspaces.py
--------------------------
Validates the SoundSpaces (Replica split) dataset and generates a human-readable
per-scene summary. Checks that all 18 scenes have the expected metadata files
(points.txt, graph.pkl) and RIR audio files for all 4 heading angles.

Outputs a summary to data/data_analysis/soundspaces_replica_summary.txt.

Usage:
    python3 data/data_preparation/01_explore_soundspaces.py
    python3 data/data_preparation/01_explore_soundspaces.py \\
        --metadata-root /path/to/SoundSpaces/metadata/replica \\
        --rir-root /path/to/SoundSpaces/binaural_rirs/replica \\
        --output-dir data/data_analysis
"""

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

ANGLES = [0, 90, 180, 270]

DEFAULT_METADATA_ROOT = "/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/metadata/replica"
DEFAULT_RIR_ROOT = "/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data_analysis"


def parse_points(points_file: Path) -> dict:
    """Parse points.txt into {index: (x, y, z)}. Format: tab-separated index x y z."""
    positions = {}
    with open(points_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                idx, x, y, z = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                positions[idx] = (x, y, z)
    return positions


def count_rir_files(rir_scene_dir: Path) -> dict:
    """
    Count RIR .wav files per angle directory.
    Returns {angle: count} for angles [0, 90, 180, 270].
    """
    counts = {}
    for angle in ANGLES:
        angle_dir = rir_scene_dir / str(angle)
        if angle_dir.exists():
            counts[angle] = sum(1 for f in angle_dir.iterdir() if f.suffix == ".wav")
        else:
            counts[angle] = None  # missing
    return counts


def explore_scene(scene_name: str, metadata_root: Path, rir_root: Path) -> dict:
    """Explore a single scene and return a stats dict."""
    meta_dir = metadata_root / scene_name
    rir_dir = rir_root / scene_name

    result = {"scene": scene_name, "errors": []}

    # --- Metadata ---
    points_file = meta_dir / "points.txt"
    graph_file = meta_dir / "graph.pkl"

    if not points_file.exists():
        result["errors"].append("MISSING: points.txt")
        result["n_positions"] = 0
    else:
        positions = parse_points(points_file)
        result["n_positions"] = len(positions)

    if not graph_file.exists():
        result["errors"].append("MISSING: graph.pkl")
        result["n_nodes"] = 0
        result["n_edges"] = 0
    else:
        with open(graph_file, "rb") as f:
            graph = pickle.load(f)
        result["n_nodes"] = graph.number_of_nodes()
        result["n_edges"] = graph.number_of_edges()

    # --- RIRs ---
    if not rir_dir.exists():
        result["errors"].append("MISSING: RIR directory")
        result["rir_counts"] = {a: None for a in ANGLES}
    else:
        result["rir_counts"] = count_rir_files(rir_dir)
        for angle, count in result["rir_counts"].items():
            if count is None:
                result["errors"].append(f"MISSING: angle dir {angle}/")

    return result


def format_summary(scenes_stats: list, metadata_root: Path, rir_root: Path) -> str:
    """Format a human-readable summary string."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 80)
    lines.append("SoundSpaces (Replica Split) — Dataset Summary")
    lines.append(f"Generated: {now}")
    lines.append(f"Metadata root: {metadata_root}")
    lines.append(f"RIR root:      {rir_root}")
    lines.append("=" * 80)
    lines.append("")

    # Column descriptions
    lines.append("Column descriptions:")
    lines.append("  Positions    : Total navigable points in the scene (from points.txt).")
    lines.append("                 These are discrete (x,y,z) locations an agent can stand at.")
    lines.append("  Graph nodes  : Positions that participate in acoustic simulation (subset of")
    lines.append("                 Positions). Only these have RIR files. From graph.pkl nodes.")
    lines.append("  Graph edges  : Navigation adjacency links between neighboring graph nodes.")
    lines.append("                 Used for pathfinding ONLY — NOT a constraint on which RIRs exist.")
    lines.append("  RIRs/angle   : Number of WAV files per heading-angle folder (0/90/180/270 deg).")
    lines.append("                 Each file = one directed (receiver, source) pair.")
    lines.append("                 Should equal graph_nodes^2 (all node-to-node pairs have RIRs).")
    lines.append("")

    # Per-scene table
    header = f"{'Scene':<25} {'Positions':>10} {'Graph nodes':>12} {'Graph edges':>12}  {'RIRs/angle (0/90/180/270)':>28}  {'Status'}"
    lines.append(header)
    lines.append("-" * len(header))

    total_positions = 0
    total_nodes = 0
    total_edges = 0
    total_rir_files = 0
    scenes_with_errors = []

    for s in scenes_stats:
        rir_counts = s.get("rir_counts", {})
        rir_str = "/".join(
            str(rir_counts.get(a, "N/A")) if rir_counts.get(a) is not None else "---"
            for a in ANGLES
        )
        status = "OK" if not s["errors"] else "ERRORS: " + "; ".join(s["errors"])
        n_pos = s.get("n_positions", 0)
        n_nodes = s.get("n_nodes", 0)
        n_edges = s.get("n_edges", 0)

        lines.append(
            f"{s['scene']:<25} {n_pos:>10} {n_nodes:>12} {n_edges:>12}  {rir_str:>28}  {status}"
        )

        total_positions += n_pos
        total_nodes += n_nodes
        total_edges += n_edges
        angle_counts = [rir_counts.get(a) for a in ANGLES if rir_counts.get(a) is not None]
        total_rir_files += sum(angle_counts) if angle_counts else 0

        if s["errors"]:
            scenes_with_errors.append(s["scene"])

    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL (18 scenes)':<25} {total_positions:>10} {total_nodes:>12} {total_edges:>12}  "
        f"{'':>28}  {total_rir_files:,} total RIR files"
    )
    lines.append("")

    # Errors section
    if scenes_with_errors:
        lines.append("SCENES WITH ERRORS:")
        for scene in scenes_with_errors:
            lines.append(f"  - {scene}")
    else:
        lines.append("All scenes validated successfully — no missing files.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Explore and validate the SoundSpaces Replica dataset.")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
        help="Path to SoundSpaces metadata/replica directory",
    )
    parser.add_argument(
        "--rir-root",
        type=Path,
        default=DEFAULT_RIR_ROOT,
        help="Path to SoundSpaces binaural_rirs/replica directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the summary text file",
    )
    args = parser.parse_args()

    metadata_root = args.metadata_root
    rir_root = args.rir_root

    if not metadata_root.exists():
        print(f"ERROR: metadata root not found: {metadata_root}")
        return
    if not rir_root.exists():
        print(f"ERROR: RIR root not found: {rir_root}")
        return

    # Discover scenes from metadata directory
    scenes = sorted(d.name for d in metadata_root.iterdir() if d.is_dir())
    print(f"Found {len(scenes)} scenes in metadata root.\n")

    scenes_stats = []
    for scene in scenes:
        print(f"  Exploring {scene}...", end=" ", flush=True)
        stats = explore_scene(scene, metadata_root, rir_root)
        scenes_stats.append(stats)
        status = "OK" if not stats["errors"] else f"{len(stats['errors'])} error(s)"
        print(status)

    summary = format_summary(scenes_stats, metadata_root, rir_root)
    print("\n" + summary)

    # Save to file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "soundspaces_replica_summary.txt"
    output_file.write_text(summary)
    print(f"Summary saved to: {output_file}")


if __name__ == "__main__":
    main()
