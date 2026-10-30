"""
Rendering an evaluation's aggregated metrics as its summary text file.

The writer takes `aggregate`'s output and an ordered dict of header fields, so the diffusion
flow and the baselines produce one layout from one implementation. Nothing is measured or
derived here beyond picking a statistic out of the aggregate: a number absent from the
aggregate prints as a dash rather than being computed on the side.

Counts are reported as failures rather than successes, against the sample total stated in the
header, so a metric that could not be measured on part of the split is visible at a glance
instead of having to be subtracted.
"""

import os
from datetime import datetime

from evaluation.lib.metrics_utils import LSD_LENGTHS

WIDTH = 100
MISSING = '—'

# The families printed as bias, MAE, RMSE, spread and percentage. A percentage of a dB-valued
# quantity has no interpretation, so those are marked and print as a dash in that column.
BROADBAND_FAMILIES = (
    ('t60',  'T60 fitted',   's',  True),
    ('t60x', 'T60 -5/-65',   's',  True),
    ('edt',  'EDT',          's',  True),
    ('c50',  'C50',          'dB', False),
    ('c80',  'C80',          'dB', False),
    ('drr',  'DRR',          'dB', False),
)

COLUMNS = (('Bias', 11), ('MAE', 10), ('RMSE', 10), ('Std', 10), ('Perc', 9), ('Fail', 7))


# =============================================================================
# Reading the aggregate
# =============================================================================

def _stat(stats: dict, key: str, which: str = 'mean'):
    """One statistic of one metric, or None when the aggregate does not carry it."""
    entry = stats.get(key)
    if isinstance(entry, dict):
        value = entry.get(which)
    elif which == 'mean':
        value = entry                      # the correlations are bare floats
    else:
        return None
    return None if value is None or value != value else float(value)


def _failed(stats: dict, key: str, n_scored: int):
    """How many samples the metric could not be measured on."""
    entry = stats.get(key)
    return None if not isinstance(entry, dict) else n_scored - int(entry.get('n', 0))


def _fmt(value, width: int, decimals: int = 3, signed: bool = False) -> str:
    """A number in its column, or a dash where there is nothing to show."""
    if value is None:
        return MISSING.rjust(width)
    return f'{value:{"+" if signed else ""}{width}.{decimals}f}'


def _fmt_count(value, width: int = 7) -> str:
    return MISSING.rjust(width) if value is None else f'{value:>{width},}'


# =============================================================================
# Blocks, in the order they are written
# =============================================================================

def _header(fields: dict, n_scored: int, n_total=None) -> list:
    """The title rule, the caller's fields, and how much of the split was scored."""
    lines = ['=' * WIDTH, 'RIR GENERATION EVALUATION'.center(WIDTH), '=' * WIDTH, '']
    pad = max(max(len(label) for label in fields), len('Samples'), len('Written')) + 2

    lines += [f'{label + ":":<{pad}}{value}' for label, value in fields.items()]
    if n_total is None or n_scored == n_total:
        scored = f'{n_scored:,} (full)' if n_total else f'{n_scored:,}'
    else:
        scored = f'{n_scored:,} of {n_total:,}'
    lines += [f'{"Samples:":<{pad}}{scored}',
              f'{"Written:":<{pad}}{datetime.now():%Y-%m-%d %H:%M:%S}', '']
    return lines


def _section(title: str) -> list:
    return ['-' * WIDTH, title, '-' * WIDTH]


def _columns() -> str:
    return f'{"metric":<16}' + ''.join(label.rjust(width) for label, width in COLUMNS)


def _score_legend() -> list:
    """What each column holds, stated as the arithmetic rather than as a description."""
    return ['  Bias  mean(gen - ref)',
            '  MAE   mean |gen - ref|',
            '  RMSE  sqrt( sum (gen - ref)^2 / N )',
            '  Perc  mean( |gen - ref| / |ref| ) x 100',
            '  Fail  samples the metric could not be measured on']


def _headline(stats: dict) -> list:
    """The five numbers a model is compared on, as a header row and a value row."""
    cells = (('T60 Perc', _fmt(_stat(stats, 't60_perc'), 8, 2) + '%'),
             ('T60 corr', _fmt(_stat(stats, 't60_corr'), 9)),
             ('LSD @min', _fmt(_stat(stats, 'lsd_min'), 9)),
             ('C50 MAE',  _fmt(_stat(stats, 'c50_abs'), 9)),
             ('DRR Abs',  _fmt(_stat(stats, 'drr_abs'), 9)))

    labels, values = [], []
    for label, value in cells:
        column = max(len(label), len(value)) + 4
        labels.append(label.rjust(column))
        values.append(value.rjust(column))
    return _section('HEADLINE') + ['', ''.join(labels), ''.join(values), '']


def _score_row(label: str, bias, mae, rmse, std, perc, fail, decimals: int) -> str:
    """One metric across the shared column set."""
    return (f'{label:<16}{_fmt(bias, COLUMNS[0][1], decimals, signed=True)}'
            f'{_fmt(mae, COLUMNS[1][1], decimals)}{_fmt(rmse, COLUMNS[2][1], decimals)}'
            f'{_fmt(std, COLUMNS[3][1], decimals)}'
            f'{_fmt(perc, COLUMNS[4][1] - 1, 2) + ("%" if perc is not None else " ")}'
            f'{_fmt_count(fail, COLUMNS[5][1])}')


def _broadband(stats: dict, n_scored: int) -> list:
    """Every scored quantity on one column set, the LSD among them."""
    lines = _section('BROADBAND METRICS') + ['', _columns()]

    for prefix, label, unit, has_perc in BROADBAND_FAMILIES:
        lines.append(_score_row(
            f'{label} ({unit})', _stat(stats, f'{prefix}_err'), _stat(stats, f'{prefix}_abs'),
            _stat(stats, f'{prefix}_err', 'rmse'), _stat(stats, f'{prefix}_err', 'std'),
            _stat(stats, f'{prefix}_perc') if has_perc else None,
            _failed(stats, f'{prefix}_abs', n_scored), 4 if unit == 's' else 3))

    lines.append(_score_row('LSD (dB)', None, _stat(stats, 'lsd_min'),
                            _stat(stats, 'lsd_min', 'rmse'), _stat(stats, 'lsd_min', 'std'),
                            None, _failed(stats, 'lsd_min', n_scored), 3))

    lines += ['', '  generated against reference across the split:']
    lines += [f'    {"corr(" + label + ")":<20}{_fmt(_stat(stats, key), 9)}'
              for label, key in (('T60 fitted', 't60_corr'), ('T60 -5/-65', 't60x_corr'))]
    return lines + [''] + _score_legend() + ['']


def _lsd_extended(stats: dict, n_scored: int) -> list:
    """Every LSD variant, each on the same column set as the scores above."""
    lines = _section('LSD EXTENDED') + ['', f'{"variant":<16}'
                                        + ''.join(label.rjust(width) for label, width in COLUMNS[1:])]

    for variant in LSD_LENGTHS:
        for name, key in (('plain', f'lsd_{variant}'), ('aligned', f'lsda_{variant}')):
            lines.append(f'{"@" + variant + " " + name:<16}'
                         f'{_fmt(_stat(stats, key), COLUMNS[1][1], 3)}'
                         f'{_fmt(_stat(stats, key, "rmse"), COLUMNS[2][1], 3)}'
                         f'{_fmt(_stat(stats, key, "std"), COLUMNS[3][1], 3)}'
                         f'{MISSING.rjust(COLUMNS[4][1])}'
                         f'{_fmt_count(_failed(stats, key, n_scored), COLUMNS[5][1])}')

    lines += ['',
              '  @gen     both shaped to the generated length, zero-padding the shorter, truncating the longer',
              '  @ref     both shaped to the native reference length, the same way',
              '  @min     both shaped to the shorter of the two',
              '  plain    no shift applied',
              "  aligned  the generated rolled so its peak index matches the reference's, zero-filled behind",
              '',
              '  Both signals are convolved with the clean speech named in the header, after shaping.',
              '  The headline uses plain @min.']
    return lines + ['']


def _per_band(stats: dict, octave_bands, n_scored: int) -> list:
    """T60 by octave band, bands across the columns so the trend reads left to right."""
    if not octave_bands:
        return []

    def band_label(fc):
        return f'{fc / 1000:g}k' if fc >= 1000 else f'{fc:g}'

    lines = _section('T60 PER OCTAVE BAND') + ['']
    lines.append(f"{'band (Hz)':<12}" + ''.join(band_label(fc).rjust(10) for fc in octave_bands))
    for label, key, which, signed in (('Bias (s)', '_err', 'mean', True),
                                      ('MAE (s)', '_abs', 'mean', False),
                                      ('RMSE (s)', '_err', 'rmse', False),
                                      ('Std (s)', '_err', 'std', False)):
        lines.append(f'{label:<12}' + ''.join(
            _fmt(_stat(stats, f't60_band_{int(fc)}{key}', which), 10, 4, signed)
            for fc in octave_bands))
    lines.append(f'{"Fail":<12}' + ''.join(
        _fmt_count(_failed(stats, f't60_band_{int(fc)}_abs', n_scored), 10)
        for fc in octave_bands))
    lines += ['', '  fitted over the same window as the broadband T60; columns as above']
    return lines + ['']


def _standalone(stats: dict, n_scored: int) -> list:
    """The metrics that have no reference counterpart to subtract from."""
    lines = _section('ALIGNMENT AND SHAPE') + ['', f'{"metric":<22}{"mean":>11}{"median":>11}'
                                               f'{"std":>11}{"Fail":>7}']
    for key, label, decimals in (('peak_shift_ms', 'Peak shift (ms)', 3),
                                 ('edc_distance', 'EDC distance', 4),
                                 ('cosine_similarity', 'Cosine similarity', 4),
                                 ('gen_length', 'Generated length', 0),
                                 ('ref_length', 'Reference length', 0)):
        lines.append(f'{label:<22}{_fmt(_stat(stats, key), 11, decimals)}'
                     f'{_fmt(_stat(stats, key, "median"), 11, decimals)}'
                     f'{_fmt(_stat(stats, key, "std"), 11, decimals)}'
                     f'{_fmt_count(_failed(stats, key, n_scored))}')
    lines += ['', '  no reference value to difference against, so these are the values themselves',
              '  peak shift is generated minus reference: positive means a late generated peak']
    return lines + ['']


# =============================================================================
# Assembling and writing
# =============================================================================

def format_summary(stats: dict, header_fields: dict, n_scored: int, n_total=None,
                   octave_bands=()) -> str:
    """
    The whole summary as one string.

    Args:
        stats:         what `aggregate` returned
        header_fields: label to value, written in order; 'Model' belongs first so every
                       flow's summary names the system it scored on its first line
        n_scored:      pairs the evaluation ran over, which every Fail column counts against
        n_total:       pairs in the split, so the header can mark a partial run; equal to
                       n_scored, or left out, prints as the whole split
        octave_bands:  the centre frequencies the per-band section covers, or empty to omit it

    Returns:
        The summary text, newline terminated.
    """
    lines = (_header(header_fields, n_scored, n_total) + _headline(stats)
             + _broadband(stats, n_scored) + _lsd_extended(stats, n_scored)
             + _per_band(stats, octave_bands, n_scored) + _standalone(stats, n_scored)
             + ['=' * WIDTH])
    return '\n'.join(lines) + '\n'


def write_summary(path: str, stats: dict, header_fields: dict, n_scored: int, n_total=None,
                  octave_bands=()) -> str:
    """
    Write the summary beside the rest of an evaluation, through a temp path and a rename.

    Returns:
        The path written.
    """
    text = format_summary(stats, header_fields, n_scored, n_total, octave_bands)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    staging = f'{path}.tmp'
    with open(staging, 'w') as handle:
        handle.write(text)
    os.replace(staging, path)
    return path
