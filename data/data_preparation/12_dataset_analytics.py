#!/usr/bin/env python3
"""
Describe what a scene dataset holds, and what training actually sees, at three stages:

    1  every pair the dataset could serve: valid, same-region, RT60 present
    2  after the quality filters and the per-room cap
    3  the train / eval / test splits

The stages differ more than they look. For Matterport3D the cap alone moves median RT60
from 0.489 s to 0.312 s, because the rooms it trims hardest are the large reverberant
ones, so a single-stage description would misstate the training distribution.

Stage 2 and the splits are built through create_soundspaces_datasets, the same code the
trainer calls, so the numbers describe the real pipeline rather than a reimplementation.
A fourth series, "capped, no filters", is drawn where it separates the cap's effect from
the filters'.

Distributions are per RIR, which is what the model sees. per_room.csv carries the
room-level view for anything that needs rooms weighted equally.

Outputs:
    data/<dataset>/analysis/per_room.csv, per_scene.csv
    data/<dataset>/analysis/figures/*.png
    docs/<dataset>_dataset_analysis.md

Requires the columns 10_precompute_rt60.py and 11_precompute_rir_metrics.py produce.

Usage:
    python3 data/data_preparation/12_dataset_analytics.py --scene-dataset matterport3d
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.soundspaces import DATASET_DEFAULTS, create_soundspaces_datasets

REQUIRED = ['rt60', 'decay_len_s', 'c50', 'drr']
STAGE_C  = {'1 all servable': '#2a6f97', '2 capped, no filters': '#f4a261',
            '2 filtered + capped': '#d00000'}
SPLIT_C  = {'train': '#2a6f97', 'eval': '#e07a00', 'test': '#8338ec'}
NOTES_RE = ('<!-- notes:start -->', '<!-- notes:end -->')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def room_key(frame, scene_dataset):
    """The room each row belongs to, aligned to the frame so it can group or map."""
    if scene_dataset == 'matterport3d':
        return pd.Series(list(zip(frame['scene'], frame['receiver_region'])), index=frame.index)
    return frame['scene']


def load_stages(scene_dataset, sample_max_sec, sr_target, mapping_csv, geometry_csv):
    """The three stages plus the no-filter diagnostic, built the way training builds them."""
    mapping = pd.read_csv(mapping_csv, comment='#')

    missing = [c for c in REQUIRED if c not in mapping.columns]
    if missing:
        raise SystemExit(
            f"{mapping_csv} is missing {', '.join(missing)}.\n"
            "Run 10_precompute_rt60.py and 11_precompute_rir_metrics.py first."
        )

    stage1 = mapping[mapping['valid'] == True]
    if scene_dataset == 'matterport3d':
        stage1 = stage1[stage1['same_region'] == True]
    stage1 = stage1[stage1['rt60'].notna()]

    common = dict(scene_dataset=scene_dataset, split=True, rir_view_type=None,
                  room_overview_type=None, use_rt60=True,
                  mapping_csv=mapping_csv, geometry_csv=geometry_csv,
                  sample_max_sec=sample_max_sec, sr_target=sr_target)
    splits   = create_soundspaces_datasets(**common)
    no_filt  = create_soundspaces_datasets(**common, min_rt60=None, max_void_indoor=None,
                                           max_void_outdoor=None, max_room_void=None)

    stages = {'1 all servable':       stage1,
              '2 capped, no filters': pd.concat([x.mapping for x in no_filt], ignore_index=True),
              '2 filtered + capped':  pd.concat([x.mapping for x in splits], ignore_index=True)}
    by_split = {n: x.mapping for n, x in zip(('train', 'eval', 'test'), splits)}
    return stages, by_split, pd.read_csv(geometry_csv, comment='#')


def with_geometry(frame, geometry, scene_dataset):
    """Attach the room's geometry to every pair, so distributions are RIR-weighted."""
    if scene_dataset == 'matterport3d':
        g = geometry.set_index(['scene', 'region_idx'])
    else:
        g = geometry.set_index(['scene'])
    key   = pd.Index(room_key(frame, scene_dataset))
    out   = frame.copy()
    for col in ('L_m', 'W_m', 'H_m', 'label', 'outdoor', 'mean_overview_void'):
        if col in g.columns:
            out[col] = key.map(g[col]).values
    out['volume_m3'] = out['L_m'] * out['W_m'] * out['H_m']
    out['aspect']    = out['L_m'] / out['W_m']
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def overlay(ax, series_by_label, colours, bins=60, xlabel='', logx=False, vline=None):
    """Normalised step histograms on one axis, so a shift shows as separated curves."""
    pooled = np.concatenate([np.asarray(v, dtype=float) for v in series_by_label.values()])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return
    edges = (np.geomspace(max(pooled.min(), 1e-3), pooled.max(), bins) if logx
             else np.linspace(pooled.min(), pooled.max(), bins))
    for label, values in series_by_label.items():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=edges, density=True, histtype='step', lw=1.7,
                    color=colours.get(label), label=f'{label}  n={len(v):,}')
    if vline is not None:
        ax.axvline(vline, color='k', ls=':', lw=1.2)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('density')
    ax.legend(fontsize=7)


def column(name):
    return lambda frame: frame[name].dropna()


def pooled_sides(frame):
    """L and W together: a room has two walls of each, and L >= W by construction,
    so plotting them apart separates the two by that definition rather than by room."""
    return pd.concat([frame['L_m'], frame['W_m']], ignore_index=True).dropna()


def two_row_figure(path, panels, stages, by_split, title, vlines=None):
    """Top row compares the stages, bottom row compares the splits."""
    vlines = vlines or {}
    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 7), squeeze=False)
    for j, (get, label) in enumerate(panels):
        overlay(axes[0][j], {k: get(v) for k, v in stages.items()},
                STAGE_C, xlabel=label, vline=vlines.get(label))
        overlay(axes[1][j], {k: get(v) for k, v in by_split.items()},
                SPLIT_C, xlabel=label, vline=vlines.get(label))
    axes[0][0].set_title('by stage', loc='left', fontsize=10, fontweight='bold')
    axes[1][0].set_title('by split', loc='left', fontsize=10, fontweight='bold')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def balance_figure(path, stages, scene_dataset):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    per_room  = {k: v.groupby(room_key(v, scene_dataset)).size() for k, v in stages.items()}
    per_scene = {k: v.groupby('scene').size() for k, v in stages.items()}
    overlay(axes[0], per_room,  STAGE_C, xlabel='RIRs per room',  logx=True)
    overlay(axes[1], per_scene, STAGE_C, xlabel='RIRs per scene', logx=True)
    for label, counts in per_room.items():
        share = np.cumsum(np.sort(counts.values)[::-1]) / counts.sum()
        axes[2].plot(100 * np.arange(1, len(share) + 1) / len(share), 100 * share,
                     lw=1.7, color=STAGE_C.get(label), label=label)
    axes[2].set_xlabel('share of rooms, largest first (%)')
    axes[2].set_ylabel('share of RIRs (%)')
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.suptitle('How the data is spread across rooms and scenes', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def relationships_figure(path, frame):
    """Physical sanity checks, and whether a condition leaks what the images should supply."""
    pairs = [('volume_m3', 'rt60',  'room volume (m3)', 'RT60 (s)',
              'Sabine check: bigger rooms should reverberate longer'),
             ('distance_m', 'rt60', 'source-receiver distance (m)', 'RT60 (s)',
              'Shortcut check: should be weak'),
             ('distance_m', 'c50',  'source-receiver distance (m)', 'C50 (dB)',
              'Control: should be strongly negative')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (xc, yc, xl, yl, note) in zip(axes, pairs):
        sub = frame[[xc, yc]].dropna()
        sub = sub[np.isfinite(sub).all(axis=1)]
        ax.hexbin(sub[xc], sub[yc], gridsize=45, bins='log', cmap='viridis', mincnt=1)
        r = np.corrcoef(sub[xc], sub[yc])[0, 1]
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'r = {r:+.2f}\n{note}', fontsize=9)
    fig.suptitle('Relationships between the conditioning signals', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables and report
# ---------------------------------------------------------------------------

def per_room_table(stages, geometry, scene_dataset):
    key_cols = ['scene', 'region_idx'] if scene_dataset == 'matterport3d' else ['scene']
    table    = geometry.copy()
    index    = pd.MultiIndex.from_frame(table[key_cols]) if len(key_cols) > 1 else pd.Index(table['scene'])
    for label, frame in stages.items():
        counts = pd.Series(1, index=pd.Index(room_key(frame, scene_dataset))).groupby(level=0).sum()
        table[f'n_rirs_{label.split()[0]}' + ('_nofilter' if 'no filters' in label else '')] = \
            index.map(counts).fillna(0).astype(int)
        if label.startswith('2 filtered'):
            med = frame.groupby(room_key(frame, scene_dataset))['rt60'].median()
            table['rt60_median'] = index.map(med)
    table['volume_m3'] = (table['L_m'] * table['W_m'] * table['H_m']).round(1)
    table['aspect']    = (table['L_m'] / table['W_m']).round(2)
    return table


def per_scene_table(stages, per_room, scene_dataset):
    rows = []
    for scene, rooms in per_room.groupby('scene'):
        row = {'scene': scene, 'rooms': len(rooms)}
        if 'outdoor' in rooms.columns:
            row['rooms_outdoor'] = int(rooms['outdoor'].sum())
        for label, frame in stages.items():
            tag = label.split()[0] + ('_nofilter' if 'no filters' in label else '')
            row[f'n_rirs_{tag}'] = int((frame['scene'] == scene).sum())
        sub = stages['2 filtered + capped']
        sub = sub[sub['scene'] == scene]
        row['rt60_median'] = round(float(sub['rt60'].median()), 3) if len(sub) else None
        rows.append(row)
    return pd.DataFrame(rows)


def highlights(stages, frame, sample_max_sec, scene_dataset):
    """Facts the tables already contain, surfaced so they are not missed."""
    s1, s2a, s2 = (stages['1 all servable'], stages['2 capped, no filters'],
                   stages['2 filtered + capped'])
    out = [f'- The per-room cap takes {len(s1):,} pairs down to {len(s2a):,}, and moves median '
           f'RT60 from {s1["rt60"].median():.3f} s to {s2a["rt60"].median():.3f} s',
           f'- The quality filters then remove {len(s2a) - len(s2):,} more pairs '
           f'({100 * (len(s2a) - len(s2)) / len(s2a):.2f}%), leaving median RT60 at '
           f'{s2["rt60"].median():.3f} s']
    long = frame['decay_len_s'].dropna()
    if len(long):
        out.append(f'- {100 * (long > sample_max_sec).mean():.2f}% of RIRs decay past '
                   f'sample_max_sec = {sample_max_sec} s')
    sub = frame[['distance_m', 'rt60']].dropna()
    out.append(f'- distance and RT60 correlate at r = '
               f'{np.corrcoef(sub["distance_m"], sub["rt60"])[0, 1]:+.2f}; a strong value would '
               f'mean the scalar conditioning already carries what the images should supply')
    sizes = s2.groupby(room_key(s2, scene_dataset)).size()
    top10 = sizes.sort_values(ascending=False).head(10).sum()
    out.append(f'- after capping, the ten largest rooms hold {100 * top10 / len(s2):.1f}% of the data')
    return out


def preserved_notes(path):
    """Keep anything the user wrote between the note markers across a regeneration."""
    start, end = NOTES_RE
    if path.exists():
        text = path.read_text()
        if start in text and end in text:
            return text.split(start)[1].split(end)[0]
    return '\n_Nothing yet. Anything written here survives regeneration._\n'


def write_report(path, scene_dataset, stages, by_split, per_room, lines, fig_dir, csv_dir):
    def table(rows, header):
        widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
        fmt    = lambda r: '| ' + ' | '.join(str(c).ljust(w) for c, w in zip(r, widths)) + ' |'
        return '\n'.join([fmt(header), '|' + '|'.join('-' * (w + 2) for w in widths) + '|',
                          *[fmt(r) for r in rows]])

    stage_rows = []
    for label, frame in stages.items():
        stage_rows.append([label, f'{len(frame):,}',
                           f'{frame.groupby(room_key(frame, scene_dataset)).ngroups:,}',
                           f'{frame["scene"].nunique():,}',
                           f'{frame["rt60"].median():.3f}'])
    split_rows = [[n, f'{len(f):,}',
                   f'{f.groupby(room_key(f, scene_dataset)).ngroups:,}',
                   f'{f["scene"].nunique():,}', f'{f["rt60"].median():.3f}']
                  for n, f in by_split.items()]

    body = [f'# {scene_dataset} dataset analysis', '',
            f'_Generated by `12_dataset_analytics.py` on {datetime.now():%Y-%m-%d %H:%M}._',
            f'_Figures: `{fig_dir}` · tables: `{csv_dir}`_', '',
            '## Notes', '', NOTES_RE[0], preserved_notes(path), NOTES_RE[1], '',
            '## Stages', '',
            table(stage_rows, ['stage', 'RIRs', 'rooms', 'scenes', 'RT60 median']), '',
            '## Splits', '',
            table(split_rows, ['split', 'RIRs', 'rooms', 'scenes', 'RT60 median']), '',
            '## Highlights', '', *lines, '',
            '## Rooms', '',
            f'- {len(per_room):,} rooms in the geometry, '
            f'{int((per_room.filter(like="n_rirs_2").iloc[:, -1] > 0).sum()):,} carrying RIRs after filtering']
    if 'outdoor' in per_room.columns:
        body.append(f'- {int(per_room["outdoor"].sum()):,} open-to-sky, '
                    f'{int((~per_room["outdoor"]).sum()):,} enclosed')
    body.append(f'- wall length pooled over L and W; aspect ratio L/W has median '
                f'{per_room["aspect"].median():.2f} and reaches {per_room["aspect"].max():.1f}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(body) + '\n')
    print(f'Report written to {path}')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene-dataset', choices=sorted(DATASET_DEFAULTS), required=True)
    ap.add_argument('--mapping-csv', default=None, help="Mapping CSV (default: the dataset's own)")
    ap.add_argument('--geometry-csv', default=None, help="Geometry CSV (default: the dataset's own)")
    ap.add_argument('--out-dir', default=None, help='Where tables and figures go')
    ap.add_argument('--report', default=None, help='Where the written report goes')
    ap.add_argument('--sample-max-sec', type=float, default=0.6,
                    help='Window training uses, drawn on the decay-length figure')
    ap.add_argument('--sr-target', type=int, default=None,
                    help='Rate training resamples to (default: the files\' own)')
    args = ap.parse_args()

    d = DATASET_DEFAULTS[args.scene_dataset]
    args.mapping_csv  = args.mapping_csv  or d['mapping_csv']
    args.geometry_csv = args.geometry_csv or d['geometry_csv']

    out_dir = Path(args.out_dir) if args.out_dir else Path(project_root) / 'data' / args.scene_dataset / 'analysis'
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading stages for {args.scene_dataset} ...')
    stages, by_split, geometry = load_stages(args.scene_dataset, args.sample_max_sec,
                                             args.sr_target, args.mapping_csv, args.geometry_csv)
    for label, frame in stages.items():
        print(f'  {label:<22} {len(frame):>10,} pairs')

    geo_stages = {k: with_geometry(v, geometry, args.scene_dataset) for k, v in stages.items()}
    geo_splits = {k: with_geometry(v, geometry, args.scene_dataset) for k, v in by_split.items()}
    final      = geo_stages['2 filtered + capped']

    print('Drawing figures ...')
    two_row_figure(fig_dir / 'rt60.png', [(column('rt60'), 'RT60 (s)')], stages, by_split,
                   'RT60 — the conditioning the model is given')
    two_row_figure(fig_dir / 'geometry.png',
                   [(pooled_sides,            'wall length (m)'),
                    (column('aspect'),        'aspect ratio L/W'),
                    (column('H_m'),           'height (m)'),
                    (column('volume_m3'),     'volume (m3)')],
                   geo_stages, geo_splits, 'Room geometry, weighted by RIR')
    two_row_figure(fig_dir / 'distance_heights.png',
                   [(column('distance_m'), 'source-receiver distance (m)')], stages, by_split,
                   'Source-receiver geometry')
    two_row_figure(fig_dir / 'decay_length.png',
                   [(column('decay_len_s'), 'time to -40 dB (s)'),
                    (column('c50'), 'C50 (dB)'), (column('drr'), 'DRR (dB)')],
                   stages, by_split, 'Measured acoustics',
                   vlines={'time to -40 dB (s)': args.sample_max_sec})
    balance_figure(fig_dir / 'balance.png', stages, args.scene_dataset)
    relationships_figure(fig_dir / 'relationships.png', final)

    print('Writing tables ...')
    per_room  = per_room_table(stages, geometry, args.scene_dataset)
    per_scene = per_scene_table(stages, per_room, args.scene_dataset)
    per_room.to_csv(out_dir / 'per_room.csv', index=False)
    per_scene.to_csv(out_dir / 'per_scene.csv', index=False)
    print(f'  per_room.csv  {len(per_room):,} rows')
    print(f'  per_scene.csv {len(per_scene):,} rows')

    report = Path(args.report) if args.report else \
        Path(project_root) / 'docs' / f'{args.scene_dataset}_dataset_analysis.md'
    write_report(report, args.scene_dataset, stages, by_split, per_room,
                 highlights(stages, final, args.sample_max_sec, args.scene_dataset), fig_dir, out_dir)


if __name__ == '__main__':
    main()
