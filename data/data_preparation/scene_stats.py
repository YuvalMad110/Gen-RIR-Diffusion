"""
Shared pieces of the mapping pipeline, so rebuilding the mapping and precomputing RT60
never discard each other's work: RT60 carries across a rebuild, and the scene table is
derived entirely from the mapping CSV, so whichever script writes it last produces a
complete table instead of dropping the columns it does not itself compute. Counts the
mapping cannot answer yet are shown as a dash.
"""

import csv
from pathlib import Path

import pandas as pd

# A WAV carrying no audio is just its 44-byte header; genuine RIRs start at ~3.4 KB.
MIN_RIR_BYTES = 1024

# What identifies a record across a rebuild: a RIR pair in the mapping, a room in the geometry.
MAPPING_KEY  = ('scene', 'receiver_idx', 'source_idx')
GEOMETRY_KEY = ('scene', 'region_idx')

_COLUMNS = {
    'replica':      ['Scene', 'Valid', 'Invalid', 'RT60 NaN', 'Total'],
    'matterport3d': ['Scene', 'Nodes', 'Regions', 'Valid', 'Invalid', 'SameReg', 'RT60 NaN', 'Total'],
}

_NOTES = {
    'replica': [
        "  - SoundSpaces samples all positions at a fixed ear height (~1.5m above floor)",
        "  - All 18 scenes are single-floor in the acoustic data",
    ],
    'matterport3d': [
        "  - Nodes     = navigable positions SoundSpaces defines for the scene, so every",
        "                ordered pair of distinct nodes is one potential RIR",
        "  - Regions   = .house regions in the scene, including any holding no navigable node",
        "  - SameReg   = receiver and source in the same .house floor polygon",
        "  - Navigable points at z ~= 1.52 m above floor (ear height)",
        "  - 85 scenes available in SoundSpaces (90 in full Matterport3D)",
    ],
}


def carry_over_columns(path, rows, columns, key=MAPPING_KEY) -> list:
    """
    Fill each row's `columns` from an existing CSV, matched on `key`. Lets a rebuilt file
    keep the columns other scripts attached to it, which cost hours to produce.

    Returns the carried columns, which the caller appends to the ones it writes.
    """
    path = Path(path)
    if not path.exists():
        return []

    wanted = set(key) | set(columns)
    prev   = pd.read_csv(path, comment='#', usecols=lambda c: c in wanted)
    found  = [c for c in columns if c in prev.columns]
    if not found:
        return []

    keys   = list(zip(*(prev[c] for c in key)))
    lookup = {c: dict(zip(keys, prev[c])) for c in found}
    for r in rows:
        k = tuple(r[c] for c in key)
        for c in found:
            value = lookup[c].get(k)
            r[c] = '' if value is None or pd.isna(value) else value
    return found


def merge_notes(header_lines, notes):
    """
    Fold note lines into a CSV comment header, replacing any earlier note about the same
    column rather than leaving both.

    A note is matched by the name before the first colon, so re-running a script with a
    different setting corrects the record instead of leaving a second line that
    contradicts it. A note spanning several lines is passed as one string with embedded
    newlines, so the whole block is replaced together.
    """
    def name_of(text):
        first = text.lstrip('#').strip().split('\n', 1)[0]
        return first.split(':', 1)[0].strip() if ':' in first else None

    merged = list(header_lines)
    for note in notes:
        key = name_of(note)
        if key is not None:
            merged = [line for line in merged if name_of(line) != key]
        block = note if note.endswith('\n') else note + '\n'
        if block not in merged:
            merged.append(block)
    return merged


def attach_columns(csv_path, key_cols, values_by_column, notes=()) -> dict:
    """
    Add columns to a CSV, matched on `key_cols`, in a single streaming pass so a multi-GB
    mapping is never held in memory and every other field keeps its exact text. Writing
    through pandas instead would reformat floats and quietly shorten precomputed columns.

    values_by_column: column name -> {key tuple: value}
    notes:            '#' lines recording how the values were produced, added once
    Returns the number of rows matched per column.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  {csv_path} not found, nothing attached")
        return {}

    columns = list(values_by_column)
    matched = {c: 0 for c in columns}
    tmp     = csv_path.with_suffix('.tmp')

    with open(csv_path, newline='') as src, open(tmp, 'w', newline='') as dst:
        line, header_lines = src.readline(), []
        while line.startswith('#'):
            header_lines.append(line)
            line = src.readline()
        dst.writelines(merge_notes(header_lines, notes))

        head = next(csv.reader([line]))
        keys = [head.index(c) for c in key_cols]
        at   = {}
        for c in columns:
            if c not in head:
                head.append(c)
            at[c] = head.index(c)
        writer = csv.writer(dst)
        writer.writerow(head)

        for row in csv.reader(src):
            if not row:
                continue
            key = tuple(row[i] for i in keys)
            row.extend([''] * (len(head) - len(row)))
            for c in columns:
                value = values_by_column[c].get(key)
                if value is not None:
                    row[at[c]] = f'{value:.4f}'
                    matched[c] += 1
            writer.writerow(row)

    tmp.replace(csv_path)
    for c in columns:
        print(f"  {c} -> {csv_path.name}: {matched[c]:,} rows matched")
    return matched


def write_scene_stats(mapping, scene_dataset, stats_path, regions_per_scene=None):
    """
    Write the per-scene table for `mapping` to `stats_path`.

    Args:
        mapping:           DataFrame of the whole mapping CSV, valid and invalid rows alike
        scene_dataset:     'replica' or 'matterport3d'
        stats_path:        destination .txt, created along with its parent
        regions_per_scene: scene -> region count, for Matterport3D's 'Regions'; regions
                           holding no navigable node never reach the mapping, so that
                           count has to come from the caller
    """
    columns  = _COLUMNS[scene_dataset]
    has_rt60 = 'rt60' in mapping.columns
    regions  = regions_per_scene or {}

    rows, totals = [], {c: 0 for c in columns[1:]}
    for scene, s in mapping.groupby('scene', sort=True):
        valid = s[s['valid'] == True]
        # Matterport3D serves only same-region pairs and RT60 covers exactly those, so a
        # cross-region row without RT60 is out of scope rather than a failed estimate
        served = valid[valid['same_region'] == True] if scene_dataset == 'matterport3d' else valid
        r = {'Scene': scene, 'Valid': len(valid), 'Invalid': len(s) - len(valid), 'Total': len(s),
             'RT60 NaN': int(served['rt60'].isna().sum()) if has_rt60 else None}
        if scene_dataset == 'matterport3d':
            r.update(Nodes=s['receiver_idx'].nunique(), Regions=regions.get(scene),
                     SameReg=int((s['same_region'] == True).sum()))
        rows.append(r)
        for c in columns[1:]:
            totals[c] = None if (r[c] is None or totals[c] is None) else totals[c] + r[c]

    widths = {c: max(len(c), 8) for c in columns[1:]}
    cell   = lambda v, w: f"{'—' if v is None else format(v, ','):>{w}}"
    line   = lambda name, r: f"{name:<28} " + " ".join(cell(r[c], widths[c]) for c in columns[1:])
    header = f"{'Scene':<28} " + " ".join(f"{c:>{widths[c]}}" for c in columns[1:])
    sep    = "-" * len(header)
    title  = f"SoundSpaces/{scene_dataset} — RIR pairs per scene"

    lines  = [title, "=" * len(title), header, sep]
    lines += [line(r['Scene'], r) for r in rows]
    lines += [sep, line('TOTAL', totals), "", "Notes:",
              "  - Each pair = one (receiver, source) directed RIR"]
    lines += _NOTES[scene_dataset]
    lines += [f"  - Invalid   = WAV file missing, or smaller than {MIN_RIR_BYTES} bytes and so holding no RIR",
              "  - RT60 NaN  = valid WAV where T60 estimation failed (EDC too short/dry); these rows",
              "                are usable for regular training but excluded from --use-rt60-condition",
              "                runs. A dash means RT60 has not been precomputed yet"]

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text("\n".join(lines) + "\n")
    print(f"Scene stats saved to: {stats_path}")
