"""
The figures an evaluation produces: per-sample comparisons, EDC per octave band, and metric
histograms over the split.

Everything drawn here is read from the per-pair rows run_evaluation.py already scored. No metric
is recomputed for a figure, so a plot can never disagree with the summary beside it. The only
quantities derived here are the curves themselves — the energy decay of a signal is not a metric
and is not stored per pair.

Each signal carries its own time axis. References are ragged by nature, so a generated RIR and
its reference rarely share a length, and one axis built from either would misplace the other.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from utils.acoustic_metrics import (DEFAULT_OCTAVE_BANDS, DEFAULT_T60_FIT_RANGE, compute_edc,
                                    compute_edc_octave_bands)

EDC_YLIM = (-60.0, 5.0)
FIT_COLOUR = 'tab:red'
REF_COLOUR = 'tab:blue'
GEN_COLOUR = 'tab:orange'

# Printed in the comparison text box, as the generated value and the signed error beside it.
HEADLINE_METRICS = (('T60', 't60', 's', '{:.3f}'),
                    ('EDT', 'edt', 's', '{:.3f}'),
                    ('C50', 'c50', 'dB', '{:.2f}'),
                    ('C80', 'c80', 'dB', '{:.2f}'),
                    ('DRR', 'drr', 'dB', '{:.2f}'))

# Histogrammed over the split. A signed error is centred on zero, a distance is not.
HISTOGRAM_METRICS = (('t60_err', 'T60 error', 's', True),
                     ('t60_perc', 'T60 error', '%', False),
                     ('edt_err', 'EDT error', 's', True),
                     ('c50_err', 'C50 error', 'dB', True),
                     ('c80_err', 'C80 error', 'dB', True),
                     ('drr_err', 'DRR error', 'dB', True),
                     ('lsd_min', 'LSD @min', 'dB', False),
                     ('peak_shift_ms', 'Peak shift', 'ms', True),
                     ('edc_distance', 'EDC distance', '', False))


# =============================================================================
# Choosing what to draw
# =============================================================================

def select_samples(rows: list, metric: str, per_category: int = 3) -> dict:
    """
    The best, median and worst pairs by one metric.

    Ranking is on the magnitude of the metric, so a signed error is ranked by how wrong it is
    rather than by its direction. A pair whose metric did not come out finite is left out of
    the ranking entirely instead of sorting to one end of it.

    Args:
        rows:         the per-pair rows run_evaluation.py scored
        metric:       the key to rank on, e.g. 'lsd_min'
        per_category: how many pairs each category holds

    Returns:
        {'best': [indices], 'median': [...], 'worst': [...]}, best first within each.
    """
    values = np.array([row.get(metric, np.nan) for row in rows], dtype=np.float64)
    usable = np.flatnonzero(np.isfinite(values))
    if usable.size == 0:
        return {}

    order = usable[np.argsort(np.abs(values[usable]))]
    count = min(per_category, order.size)
    middle = order.size // 2
    start = max(0, middle - count // 2)

    return {'best': order[:count].tolist(),
            'median': order[start:start + count].tolist(),
            'worst': order[-count:][::-1].tolist()}


def format_title(parts: dict) -> str:
    """
    A title from the parts that are present, so a missing one prints nothing at all.

    Args:
        parts: label to value, in order; a value of None or '' is dropped

    Returns:
        The parts joined with a separator, labels included only where given.
    """
    pieces = [f'{label}: {value}' if label else str(value)
              for label, value in parts.items() if value not in (None, '')]
    return '  ·  '.join(pieces)


# =============================================================================
# Shared drawing
# =============================================================================

def _axis_limit(sr: int, max_sec=None, *signals) -> float:
    """The time axis' right edge in seconds: the longest signal, or the clamp when it is shorter."""
    longest = max(len(np.asarray(s).flatten()) for s in signals) / sr
    return min(longest, max_sec) if max_sec else longest


def _clip_note(signal, sr: int, label: str, max_sec=None) -> str:
    """
    How long a signal is, saying so explicitly when the axis hides part of it.

    The note is appended to a title that is already drawn, so it changes the text and not the
    layout.
    """
    seconds = len(np.asarray(signal).flatten()) / sr
    if max_sec and seconds > max_sec:
        return f'{label} {seconds:.2f} s (axis clipped to {max_sec:.2f} s)'
    return f'{label} {seconds:.2f} s'


def _plot_waveform(ax, signal, sr: int, colour: str, title: str, xlim: float) -> None:
    """One waveform on its own time axis."""
    signal = np.asarray(signal, dtype=np.float64).flatten()
    ax.plot(np.arange(len(signal)) / sr, signal, lw=0.5, color=colour)
    ax.set_xlim(0, xlim)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('s', fontsize=7)
    ax.tick_params(labelsize=6)


def _plot_overlay(ax, reference, generated, sr: int, xlim: float) -> None:
    """
    Both waveforms on one axis, each scaled to its own peak.

    The two differ in absolute level by a factor of several, so an unscaled overlay draws the
    quieter one as a flat line. The absolute levels are already visible in the two waveform
    panels; this one is for the shape.
    """
    for signal, colour in ((reference, REF_COLOUR), (generated, GEN_COLOUR)):
        signal = np.asarray(signal, dtype=np.float64).flatten()
        peak = np.abs(signal).max()
        ax.plot(np.arange(len(signal)) / sr, signal / peak if peak > 0 else signal,
                lw=0.5, color=colour, alpha=0.75)

    ax.set_xlim(0, xlim)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title('overlay, each scaled to its own peak', fontsize=8)
    ax.set_xlabel('s', fontsize=7)
    ax.tick_params(labelsize=6)


def _draw_fit_window(ax, fit_range) -> None:
    """Mark the two levels the T60 is fitted between."""
    for level in fit_range:
        ax.axhline(level, color=FIT_COLOUR, ls=':', lw=0.8, alpha=0.7)


def _plot_edc(ax, reference, generated, sr: int, xlim: float, fit_range=None) -> None:
    """Both energy decay curves, each on its own time axis."""
    reference_edc = compute_edc(np.asarray(reference, dtype=np.float64).flatten())
    generated_edc = compute_edc(np.asarray(generated, dtype=np.float64).flatten())

    ax.plot(np.arange(len(reference_edc)) / sr, reference_edc, lw=0.9, color=REF_COLOUR, label='reference')
    ax.plot(np.arange(len(generated_edc)) / sr, generated_edc, lw=0.9, color=GEN_COLOUR, label='generated')
    if fit_range is not None:
        _draw_fit_window(ax, fit_range)

    ax.set_xlim(0, xlim)
    ax.set_ylim(*EDC_YLIM)
    ax.set_title('EDC', fontsize=8)
    ax.set_xlabel('s', fontsize=7)
    ax.set_ylabel('dB', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc='upper right')


def _metric_lines(row: dict) -> list:
    """
    The headline metrics as the reference value and the signed error against it.

    The generated value is the two added, so showing it as well would spend a column on a
    number already on the line.
    """
    lines = [f'{"METRICS":<6}{"ref":>12}{"err":>11}', '']
    for label, prefix, unit, spec in HEADLINE_METRICS:
        reference, error = row.get(f'{prefix}_ref'), row.get(f'{prefix}_err')
        shown = (f'{spec.format(reference)} {unit}'
                 if reference is not None and np.isfinite(reference) else '—')
        signed = f'{error:+.3f}' if error is not None and np.isfinite(error) else '—'
        lines.append(f'{label:<6}{shown:>12}{signed:>11}')

    lsd = row.get('lsd_min')
    shown = f'{lsd:.3f} dB' if lsd is not None and np.isfinite(lsd) else '—'
    lines += ['', f'{"LSD":<6}{shown:>12}']
    return lines


def _scenario_lines(row: dict, room_dim=None) -> list:
    """What the model was asked for, which is the same for every metric it is judged on."""
    lines = ['SCENARIO PARAMS', '']
    lines.append(f"scene     {row.get('scene', '—')}")
    lines.append(f"room      {row.get('room_id', '—')}  ({row.get('room_type', '—')})")
    if room_dim is not None:
        lines.append('dims      ' + ' x '.join(f'{d:.1f}' for d in np.asarray(room_dim).flatten()) + ' m')

    rt60, distance = row.get('t60_ref'), row.get('distance_m')
    lines.append(f'RT60      {rt60:.3f} s' if rt60 is not None and np.isfinite(rt60) else 'RT60      —')
    lines.append(f'distance  {distance:.2f} m' if distance is not None else 'distance  —')
    return lines


def _plot_text_box(ax, row: dict, room_dim=None, fontsize: float = 9.0) -> None:
    """The two text columns inside one frame, metrics on the left and the scenario on the right."""
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle='round,pad=0.015',
                                transform=ax.transAxes, facecolor='#f6f6f4',
                                edgecolor='#9a9a9a', lw=0.9, zorder=0))

    for x, lines in ((0.05, _metric_lines(row)), (0.50, _scenario_lines(row, room_dim))):
        ax.text(x, 0.90, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left',
                fontsize=fontsize, family='monospace', zorder=1)


# =============================================================================
# The three figures
# =============================================================================

def plot_comparison(selected: dict, generations: dict, references: dict, rows: list, sr: int,
                    title: str, save_path, max_sec=None, fit_range=DEFAULT_T60_FIT_RANGE):
    """
    One row per selected pair: the two waveforms, their overlay, the EDC and the text box.

    Args:
        selected:    what select_samples returned
        generations: the loaded generation set
        references:  the loaded reference store
        rows:        the per-pair rows, indexed as the stores are
        sr:          sample rate
        title:       the figure's title, already assembled
        save_path:   where the png goes
        max_sec:     clamp for every time axis, or None to show each signal whole
        fit_range:   the T60 fit window to mark, or None to leave the EDC unmarked

    Returns:
        save_path.
    """
    ordered = [(category, index) for category in ('best', 'median', 'worst')
               for index in selected.get(category, [])]
    if not ordered:
        raise ValueError('nothing selected to plot')

    # The text panel holds two monospace columns, so it needs more width than a plot does
    fig, axes = plt.subplots(len(ordered), 5, figsize=(23.5, 2.6 * len(ordered)), squeeze=False,
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.7]})

    for row_axes, (category, index) in zip(axes, ordered):
        reference = np.asarray(references['references'][index], dtype=np.float64).flatten()
        generated = np.asarray(generations['generated'][index], dtype=np.float64).flatten()
        xlim = _axis_limit(sr, max_sec, reference, generated)

        _plot_waveform(row_axes[0], reference, sr, REF_COLOUR,
                       f'{category} · {rows[index].get("sample_id", index)}\n'
                       + _clip_note(reference, sr, 'ref', max_sec), xlim)
        _plot_waveform(row_axes[1], generated, sr, GEN_COLOUR,
                       _clip_note(generated, sr, 'gen', max_sec), xlim)

        _plot_overlay(row_axes[2], reference, generated, sr, xlim)
        _plot_edc(row_axes[3], reference, generated, sr, xlim, fit_range)
        _plot_text_box(row_axes[4], rows[index], references['room_dim'][index])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_edc_per_band(selected: dict, generations: dict, references: dict, rows: list, sr: int,
                      title: str, save_path, octave_bands=DEFAULT_OCTAVE_BANDS, max_sec=None,
                      fit_range=DEFAULT_T60_FIT_RANGE):
    """
    One panel per selected pair and octave band, the reference decay against the generated one.

    Returns:
        save_path.
    """
    ordered = [(category, index) for category in ('best', 'median', 'worst')
               for index in selected.get(category, [])]
    if not ordered:
        raise ValueError('nothing selected to plot')

    fig, axes = plt.subplots(len(ordered), len(octave_bands),
                             figsize=(3.1 * len(octave_bands), 2.5 * len(ordered)), squeeze=False)

    for row_axes, (category, index) in zip(axes, ordered):
        reference = np.asarray(references['references'][index], dtype=np.float64).flatten()
        generated = np.asarray(generations['generated'][index], dtype=np.float64).flatten()
        xlim = _axis_limit(sr, max_sec, reference, generated)

        reference_bands = compute_edc_octave_bands(reference, sr, octave_bands)
        generated_bands = compute_edc_octave_bands(generated, sr, octave_bands)

        for ax, centre in zip(row_axes, octave_bands):
            reference_edc, generated_edc = reference_bands.get(centre), generated_bands.get(centre)
            if reference_edc is not None:
                ax.plot(np.arange(len(reference_edc)) / sr, reference_edc, lw=0.9,
                        color=REF_COLOUR, label='reference')
                if fit_range is not None:
                    _draw_fit_window(ax, fit_range)
            if generated_edc is not None:
                ax.plot(np.arange(len(generated_edc)) / sr, generated_edc, lw=0.9,
                        color=GEN_COLOUR, label='generated')

            ax.set_xlim(0, xlim)
            ax.set_ylim(*EDC_YLIM)
            ax.set_title(f'{int(centre)} Hz', fontsize=8)
            ax.tick_params(labelsize=6)

        row_axes[0].set_ylabel(f'{category}\n{rows[index].get("sample_id", index)}\ndB', fontsize=7)

    axes[0][0].legend(fontsize=6, loc='upper right')
    for ax in axes[-1]:
        ax.set_xlabel('s', fontsize=7)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_histograms(rows: list, title: str, save_path, bins: int = 40):
    """
    The distribution of each metric over the split, signed errors marked at zero.

    A metric with no finite value is drawn as an empty panel saying so, rather than being
    dropped, so the grid keeps the same shape between models.

    Returns:
        save_path.
    """
    columns = 3
    panels = len(HISTOGRAM_METRICS)
    layout = (-(-panels // columns), columns)
    fig, axes = plt.subplots(*layout, figsize=(5.2 * columns, 3.2 * layout[0]), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, (key, label, unit, signed) in zip(flat, HISTOGRAM_METRICS):
        values = np.array([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]

        if finite.size == 0:
            ax.text(0.5, 0.5, f'{label}\nno finite values', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.hist(finite, bins=bins, color=GEN_COLOUR, edgecolor='white', linewidth=0.4)
        if signed:
            ax.axvline(0.0, color=FIT_COLOUR, ls='--', lw=1.0)
        ax.axvline(finite.mean(), color='black', ls='-', lw=1.0,
                   label=f'mean {finite.mean():.3f}')

        ax.set_title(f'{label}{f" [{unit}]" if unit else ""}  ·  n {finite.size:,}', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    for ax in flat[panels:]:
        ax.axis('off')

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
