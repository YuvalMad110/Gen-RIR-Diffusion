#!/usr/bin/env python3
"""
Compare the ways of reading an RIR's decay rate, on both of the jobs that rate does:
flattening the RIR for training, and estimating RT60 for the conditioning vector.

Both come from the same number. Fitting a straight line to the energy decay curve gives
a slope in dB/s; from it,

    k   = -slope * ln(10) / 20     the rate scale_rir multiplies by, as exp(k*t)
    T60 = 60 / -slope              the seconds that line needs to fall 60 dB

so a window that describes the decay well gives both a flat scaled signal and a
trustworthy RT60. The windows differ only in which stretch of the decay curve they fit.

One window serves both jobs, because the scaling rate is derived from the stored
reverberation time rather than refitted at training time. That fixes where each half is
measured: the decay is fitted to the recording at its native length, since reverberation
time is computed once in data preparation, while the scaling is judged on the cropped and
padded signal training actually serves. Nothing here applies a zero tail — the point of
choosing a better window is to stop deleting the decay the model should be learning.

Scaling quality
    scale_rir multiplies by exp(k*t) to remove the exponential decay, so the model sees a
    roughly stationary signal. If k is right the scaled envelope is flat; too small leaves
    decay behind, too large makes the signal grow, which is what drove values past the
    half-precision ceiling and produced NaN losses. The envelope is the thing to measure,
    not the decay curve: a perfectly flattened RIR is stationary noise, whose decay curve
    is 10*log10(1 - t/T) rather than a straight line. So the metric is the slope of the
    framed RMS envelope in dB/s, measured from the -5 dB point to the end of the recorded
    signal. Zero is perfect, negative means under-correction, positive over-correction.

RT60 quality
    There is no ground-truth RT60 — SoundSpaces ships the RIRs and nothing else — so
    quality is judged three ways:

    accuracy     Every window is scored against the slope of the envelope over the late
                 part of the signal, which no window is fitted to. The envelope is the
                 reference rather than the decay curve because these recordings are cut
                 off while still ringing: the decay curve integrates the energy still to
                 come, so it falls vertically wherever the file ends, while the envelope
                 simply stops. The report prints how much decay each file actually holds,
                 and it is less than the depths these windows reach. In dB the envelope
                 and the decay curve carry the same slope, since one is 10*log10 of energy
                 and the other 20*log10 of amplitude, so comparing a window's slope
                 against the envelope's is fair.
    consistency  T60 is a property of the room, so every pair measured in one room should
                 read the same. Within-room spread and the within-room correlation with
                 source-receiver distance both say how far from that an estimator falls.
                 This criterion is untouched by truncation, which makes it the firmest.
    coverage     How many RIRs the window can be fitted on at all. A window reaching to
                 -40 dB has nothing to fit on an RIR that never gets there.

Reading the RIRs is the whole cost, at roughly 20 ms each over the network filesystem,
so --n-rirs 0 takes about six hours for Matterport3D. A random sample of tens of
thousands separates the methods by factors rather than percentages, so sampling is the
intended mode and the full pass buys precision the comparison does not need.

Usage:
    python3 data/data_preparation/13_scaling_check.py --scene-dataset matterport3d
    python3 data/data_preparation/13_scaling_check.py --scene-dataset matterport3d \\
        --n-rirs 100000 --out-dir data/matterport3d/analysis/scaling_windows
"""

import argparse
import os
import sys
import textwrap
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.acoustic_metrics import compute_edc
from utils.signal_proc import waveform_to_spectrogram
from data.soundspaces import DATASET_DEFAULTS, _room_keys, _subsample_rooms

START_DB        = -5.0            # judge the flattening from here, past the direct sound
LATE_START_DB   = -25.0           # the reference region begins here, below every window's
                                  # disagreement and deep in the tail the model has to learn
FRAME_MS        = 10.0
MIN_LATE_FRAMES = 8               # frames the reference slope needs to mean anything
FP16_MAX        = 65504.0
DIAGNOSTIC_DB   = (-30.0, -40.0)  # crossings recorded only to show how much decay a file holds
NOTES_RE        = ('<!-- notes:start -->', '<!-- notes:end -->')
MIN_ROOM_PAIRS  = 10              # rooms with fewer measurements are left out of the
                                  # within-room statistics, where a handful of pairs would
                                  # describe the sample rather than the estimator

# Each window fits the decay over a different stretch of the decay curve, in dB. The four
# combinations of a -10 or -15 top edge with a -35 or -40 bottom edge bracket the best
# performers measured so far; -10..-30 carries the bottom-edge trend one step shallower.
WINDOWS = {
    'win -10..-40': (-10.0, -40.0),
    'win -15..-40': (-15.0, -40.0),
    'win -15..-35': (-15.0, -35.0),
    'win -10..-30': (-10.0, -30.0),
}
VARIANTS = tuple(WINDOWS)
COLOURS  = {'win -10..-40': '#2a9d8f', 'win -15..-40': '#0b525b',
            'win -15..-35': '#2a6f97', 'win -10..-30': '#e07a00'}

# Every dB level any window starts or ends at, which is what window_edges_figure marks on
# the waveform so the windows can be judged against the signal they are cut from.
WINDOW_EDGES = tuple(sorted({edge for window in WINDOWS.values() for edge in window},
                            reverse=True))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def fit_window(edc_db, sr, hi, lo):
    """
    Least-squares line through the stretch of the decay curve lying between hi and lo dB.
    Returns its slope in dB/s and its intercept in dB, so the line can be both measured
    and drawn. Fewer than three samples in range is no fit.
    """
    idx = np.flatnonzero((edc_db <= hi) & (edc_db >= lo))
    if idx.size < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(idx / sr, edc_db[idx], 1)
    return float(slope), float(intercept)


def k_from_slope(slope):
    """The decay curve is energy, so its dB slope is -20k/ln(10); inverting that gives k."""
    return -slope * np.log(10) / 20


def t60_from_slope(slope):
    """Seconds a line of this slope needs to fall 60 dB."""
    return 60.0 / -slope if np.isfinite(slope) and slope < 0 else np.nan


def crossing_time(edc_db, sr, depth):
    """Seconds until the decay curve first falls to depth dB, NaN if it never does."""
    below = np.flatnonzero(edc_db <= depth)
    return float(below[0] / sr) if below.size else np.nan


def decay_fits(full, sr):
    """
    The fitted line for each window, and the decay curve they were fitted to.

    The fit is taken on the recording at its native length. Cropping first would move the
    point where the decay curve runs out of signal, and a window reaching below that point
    would then be fitting the recording's end rather than the room — which is how a window
    comes to report a reverberation time of a twentieth of a second. Reverberation time is
    computed once in data preparation, where the full recording is available, so there is
    nothing to gain by measuring it on the shorter signal training happens to serve.
    """
    edc_db = compute_edc(full)                     # already in dB
    fits   = {name: fit_window(edc_db, sr, hi, lo) for name, (hi, lo) in WINDOWS.items()}
    return fits, edc_db


def scale(rir, sr, k):
    t = np.arange(len(rir)) / sr
    return rir * np.exp(np.clip(k * t, None, 700))   # clip only to keep numpy finite


def envelope_db(signal, sr, start, end):
    """Framed RMS envelope in dB, and frame times, over the region being judged."""
    frame = max(1, int(sr * FRAME_MS / 1000))
    first, last = int(start) // frame, int(end) // frame
    if last - first < 3:
        return None, None
    seg = signal[first * frame:last * frame].reshape(last - first, frame)
    db  = 20 * np.log10(np.sqrt((seg ** 2).mean(axis=1)) + 1e-12)
    return (np.arange(first, last) + 0.5) * frame / sr, db


def envelope_slope(signal, sr, start, end, min_frames=3):
    """dB per second of the envelope over a region, and its peak-to-trough span."""
    t, db = envelope_db(signal, sr, start, end)
    if t is None:
        return np.nan, np.nan
    ok = np.isfinite(db) & (db > -200)
    if ok.sum() < min_frames:
        return np.nan, np.nan
    return float(np.polyfit(t[ok], db[ok], 1)[0]), float(db[ok].max() - db[ok].min())


def late_reference(rir, sr, edc_db, end):
    """
    Slope of the envelope over the late part of the signal, in dB/s — the decay rate every
    window is scored against.

    It runs from the LATE_START_DB point of the decay curve to the end of the recording,
    which is below where the windows start disagreeing and deep in the tail the model has
    to reproduce. No window is fitted here, so every one of them is extrapolating onto it.
    """
    after = np.flatnonzero(edc_db <= LATE_START_DB)
    if not after.size:
        return np.nan
    slope, _ = envelope_slope(rir, sr, int(after[0]), end, min_frames=MIN_LATE_FRAMES)
    return slope


def measure(full, served, sr, end):
    """
    Every per-window number for one RIR, and the per-RIR reference they are scored on.

    The decay is read from `full`, the recording at its native length, because that is
    where reverberation time is measured in data preparation. The scaling is then applied
    to `served`, the cropped and padded signal training feeds the model, and the leftover
    drift is judged over its real samples only. Uncropped in, cropped out — which is
    exactly the path a filter built in preparation has to reproduce.
    """
    fits, edc_full = decay_fits(full, sr)
    t60_late = t60_from_slope(late_reference(full, sr, edc_full, len(full)))

    edc_served = compute_edc(served)
    after      = np.flatnonzero(edc_served <= START_DB)
    start      = int(after[0]) if after.size else 0

    shared = {'rec_len_s': len(full) / sr, 'served_len_s': end / sr, 't60_late': t60_late}
    shared.update({f'obs{int(-depth)}': crossing_time(edc_full, sr, depth)
                   for depth in DIAGNOSTIC_DB})

    out = {}
    for name, (slope, _) in fits.items():
        rec = {'k': np.nan, 'drift': np.nan, 'span': np.nan, 'peak': np.nan,
               't60': np.nan, 't60_err_pct': np.nan}
        if np.isfinite(slope) and slope < 0:
            k      = k_from_slope(slope)
            scaled = scale(served, sr, k)
            rec['k'] = k
            rec['drift'], rec['span'] = envelope_slope(scaled, sr, start, end)
            rec['peak'] = float(waveform_to_spectrogram(torch.tensor(scaled)[None, :],
                                                        hop_length=64, n_fft=256).abs().max())
            rec['t60'] = t60_from_slope(slope)
            if np.isfinite(t60_late) and t60_late > 0:
                rec['t60_err_pct'] = 100 * (rec['t60'] - t60_late) / t60_late
        out[name] = rec
    return out, shared


def load_rir(root, row, sr_target, sample_max_sec):
    """
    The recording twice: at its native length, and as training serves it.

    The native-length signal is what the decay is fitted to, since reverberation time is
    computed in data preparation where the whole recording is in hand. The served signal
    is the same audio cropped and zero-padded to sample_max_sec, which is what the model
    is given and therefore what the scaling has to flatten. `end` is the last real sample
    inside the served signal, so the padding is never judged as if it were signal.
    """
    waveform, sr = torchaudio.load(os.path.join(root, row.rir_path))
    rir = waveform[int(row.best_channel):int(row.best_channel) + 1]
    if sr_target and sr != sr_target:
        rir, sr = torchaudio.transforms.Resample(sr, sr_target)(rir), sr_target

    full   = rir[0].numpy()
    length = int(sr * sample_max_sec)
    served = torch.nn.functional.pad(rir, (0, max(0, length - rir.shape[-1])))[0, :length]
    return full, served.numpy(), sr, min(len(full), length)


def room_label(row, scene_dataset):
    """The room a pair sits in, as one string, so it can group the within-room statistics."""
    if scene_dataset == 'matterport3d':
        return f'{row.scene}|{row.receiver_region}'
    return str(row.scene)


def within_room(frame, column):
    """
    How much `column` varies inside a single room, and how much of that variation tracks
    source-receiver distance.

    T60 belongs to the room, so an estimator that reads differently for two pairs in the
    same room is responding to the pair as well. Spread is the pooled within-room standard
    deviation: the square root of the mean per-room variance. The correlation is taken
    after subtracting each room's own mean from both columns, which removes room identity
    and leaves only the variation inside rooms — it should be zero.
    """
    sub = frame[['room', 'distance_m', column]].dropna()
    sub = sub[sub.groupby('room')[column].transform('size') >= MIN_ROOM_PAIRS]
    if len(sub) < MIN_ROOM_PAIRS:
        return np.nan, np.nan, 0
    variance = sub.groupby('room')[column].var(ddof=1).dropna()
    dev_val  = sub[column]       - sub.groupby('room')[column].transform('mean')
    dev_dist = sub['distance_m'] - sub.groupby('room')['distance_m'].transform('mean')
    coupling = (float(np.corrcoef(dev_val, dev_dist)[0, 1])
                if dev_val.std() > 0 and dev_dist.std() > 0 else np.nan)
    return float(np.sqrt(variance.mean())), coupling, int(variance.size)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def examples_figure(examples, path):
    """Original and scaled envelopes side by side, so flatness is visible not inferred."""
    def original(ax, ex):
        t, db = envelope_db(ex['rir'], ex['sr'], ex['start'], ex['end'])
        ax.plot(1000 * t, db, lw=1.4, color='k')

    def scaled(ax, ex, k):
        ts, dbs = envelope_db(scale(ex['rir'], ex['sr'], k), ex['sr'], ex['start'], ex['end'])
        ok      = np.isfinite(dbs) & (dbs > -200)
        ax.plot(1000 * ts, dbs, lw=1.4, color='#2a6f97')
        if ok.sum() < 3:
            return f'k={k:.0f}'
        line = np.polyfit(ts[ok], dbs[ok], 1)
        ax.plot(1000 * ts, np.polyval(line, ts), 'k--', lw=1)
        return f'k={k:.0f}  drift {line[0]:+.0f} dB/s'

    _example_grid(examples,
                  'Envelope before and after scaling — a flat line means the decay was removed',
                  path, original, scaled, share_row_scale=True)


def _example_grid(examples, title, path, draw_original, draw_scaled, share_row_scale=False):
    """
    The shared layout for the three before-and-after figures: one row per RIR, the
    original in the first column and each window's scaled result beside it.

    With share_row_scale the scaled panels in a row are put on one vertical scale, so the
    windows can be compared to each other by eye rather than each being autoscaled to its
    own range. The original keeps its own scale, being a different size of signal.
    """
    cols = len(VARIANTS) + 1
    fig, axes = plt.subplots(len(examples), cols, figsize=(4.2 * cols, 2.9 * len(examples)),
                             squeeze=False)
    for r, ex in enumerate(examples):
        draw_original(axes[r][0], ex)
        axes[r][0].set_ylabel(f'RT60 {ex["rt60"]:.2f}s\n{ex["distance_m"]:.1f} m apart', fontsize=8)
        axes[r][0].set_title('original  (own scale)' if r == 0 else '', fontsize=10,
                             fontweight='bold')
        drawn = []
        for c, name in enumerate(VARIANTS, start=1):
            ax    = axes[r][c]
            slope = ex['fits'][name][0]
            if not (np.isfinite(slope) and slope < 0):
                ax.text(.5, .5, 'no estimate', ha='center', transform=ax.transAxes)
                continue
            heading = draw_scaled(ax, ex, k_from_slope(slope))
            ax.set_title(f'{name}  {heading}' if r == 0 else heading, fontsize=9)
            drawn.append(ax)
        if share_row_scale and drawn:
            low  = min(ax.get_ylim()[0] for ax in drawn)
            high = max(ax.get_ylim()[1] for ax in drawn)
            for ax in drawn:
                ax.set_ylim(low, high)
        for ax in axes[r]:
            ax.set_xlabel('time (ms)' if r == len(examples) - 1 else '')
            ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def waveform_examples_figure(examples, path):
    """
    The signal itself, before and after scaling, over the region the flattening is judged on.

    This is what the model is handed, so it shows what the envelope smooths away: whether
    the late samples swell once the decay is divided out. The scaled panels in a row share
    one vertical scale so the windows can be compared directly; the original keeps its
    own, being far smaller. The heading carries the spectrogram peak, because the
    half-precision ceiling applies to the spectrogram rather than to these sample values.
    """
    def region(ex):
        return ex['rir'][ex['start']:ex['end']], 1000 * np.arange(ex['start'], ex['end']) / ex['sr']

    def original(ax, ex):
        signal, t_ms = region(ex)
        ax.plot(t_ms, signal, lw=0.4, color='k')

    def scaled(ax, ex, k):
        whole = scale(ex['rir'], ex['sr'], k)
        t_ms  = 1000 * np.arange(ex['start'], ex['end']) / ex['sr']
        ax.plot(t_ms, whole[ex['start']:ex['end']], lw=0.4, color='#2a6f97')
        peak = float(waveform_to_spectrogram(torch.tensor(whole)[None, :],
                                             hop_length=64, n_fft=256).abs().max())
        return f'spectrogram peak {peak:,.0f}' + ('  OVER fp16' if peak > FP16_MAX else '')

    _example_grid(examples,
                  'Waveform before and after scaling — a swelling tail is what overflows fp16',
                  path, original, scaled, share_row_scale=True)


def edc_examples_figure(examples, path):
    """
    The decay curve before and after scaling, against the curve a perfect result would give.

    A fully flattened RIR is stationary noise, and the decay curve of stationary noise is
    10*log10(1 - t/T) over the region measured, not a flat line: the backward integral
    runs out of signal even when nothing is decaying. The dashed grey curve is that
    target, so the scaled panel is read as how closely it tracks the target rather than
    how flat it is.
    """
    def ideal(n):
        u = np.arange(n) / n
        with np.errstate(divide='ignore'):
            return 10 * np.log10(np.maximum(1 - u, 1e-12))

    def original(ax, ex):
        segment = ex['rir'][ex['start']:ex['end']]
        t_ms    = 1000 * np.arange(ex['start'], ex['end']) / ex['sr']
        ax.plot(t_ms, compute_edc(segment), lw=1.4, color='k')
        ax.set_ylim(-60, 3)

    def scaled(ax, ex, k):
        segment = scale(ex['rir'], ex['sr'], k)[ex['start']:ex['end']]
        t_ms    = 1000 * np.arange(ex['start'], ex['end']) / ex['sr']
        curve   = compute_edc(segment)
        target  = ideal(len(segment))
        ax.plot(t_ms, target, lw=1.2, ls='--', color='#888888')
        ax.plot(t_ms, curve,  lw=1.4, color='#2a6f97')
        ax.set_ylim(-60, 3)
        inside = slice(0, int(0.9 * len(curve)))       # the last tenth is the integral running out
        return f'gap to target {np.nanmedian(curve[inside] - target[inside]):+.1f} dB'

    _example_grid(examples,
                  'Decay curve before and after scaling — dashed grey is what a perfect result gives',
                  path, original, scaled)


def window_edges_figure(examples, path, ncols=2):
    """
    The waveform itself, for RIRs spread across the RT60 range, with every window edge
    marked at the moment it falls.

    The windows are stated in dB on the decay curve, which says nothing about where they
    land on the signal. Drawing each edge at the time the decay curve passes it shows how
    much of the direct sound a window still contains, and how much of the reverberant tail
    it reaches. The framed RMS envelope is traced over the waveform in both directions,
    since the bare waveform is too dense to read a decay from.

    The vertical scale is set from the reverberant part rather than the peak, so the
    direct sound runs off the top of the panel. At full scale that one spike is tens of
    times anything after it and flattens the whole tail into a line, which is the part the
    windows are being judged on.
    """
    nrows = int(np.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, 2.3 * nrows), squeeze=False)
    for ax in axes.ravel()[len(examples):]:
        ax.axis('off')

    shades = plt.cm.plasma(np.linspace(0.05, 0.85, len(WINDOW_EDGES)))
    for ax, ex in zip(axes.ravel(), examples):
        sr, full = ex['sr'], ex['full']
        amp      = full / np.abs(full).max()
        t_ms     = 1000 * np.arange(len(full)) / sr
        ax.plot(t_ms, amp, lw=0.4, color='#9ec5e8')

        t_env, db_env = envelope_db(full, sr, 0, len(full))
        if t_env is not None:
            env = 10 ** (db_env / 20) / np.abs(full).max()
            ax.plot(1000 * t_env,  env, lw=1.3, color='#1f4e79')
            ax.plot(1000 * t_env, -env, lw=1.3, color='#1f4e79')
        if ex['end'] < len(full):
            ax.axvline(1000 * ex['end'] / sr, color='#d00000', ls='-.', lw=1.2)

        # scale from the reverberation, not the direct peak, which is far larger
        reverb = amp[int(sr * 0.04):]
        limit  = min(1.05, 6 * float(np.sqrt((reverb ** 2).mean()))) if reverb.size else 1.05

        for row, (colour, depth) in enumerate(zip(shades, WINDOW_EDGES)):
            t = crossing_time(ex['edc'], sr, depth)
            if not np.isfinite(t):
                continue
            ax.axvline(1000 * t, color=colour, lw=1.1, ls='--', alpha=0.9)
            ax.text(1000 * t, limit * (1.06 + 0.22 * (row % 2)), f'{int(depth)}',
                    color=colour, fontsize=7, ha='center', va='bottom')

        ax.set_ylim(-limit * 1.1, limit * 1.45)
        ax.set_xlim(0, t_ms[-1] if len(full) else 1)
        ax.set_ylabel(f'RT60 {ex["rt60"]:.2f}s', fontsize=8)
        ax.set_xlabel('time (ms)', fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle('Where the window edges fall on the signal — labels are dB on the decay curve.\n'
                 'Vertical scale is set by the reverberation, so the direct sound runs off the panel',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def decay_fits_figure(examples, path, ncols=3):
    """
    One panel per RIR: the decay curve with every window's fitted line drawn over it.

    Each line is thick across the stretch it was fitted to and dashed beyond it, so a
    window's fit can be told from its extrapolation. The envelope sits behind in grey, and
    in dB it carries the same slope as the decay curve, so a line that follows one follows
    the other. Where the two part company near the end of the panel, the decay curve is
    falling because the recording stops, not because the room has gone quiet — which is
    why the late reference slope is taken from the envelope.
    """
    nrows = int(np.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.3 * nrows), squeeze=False)
    for ax in axes.ravel()[len(examples):]:
        ax.axis('off')

    for ax, ex in zip(axes.ravel(), examples):
        sr, full = ex['sr'], ex['full']
        edc      = ex['edc']
        t_ms     = 1000 * np.arange(len(full)) / sr

        t_env, db_env = envelope_db(full, sr, 0, len(full))
        if t_env is not None:
            ax.plot(1000 * t_env, db_env - np.nanmax(db_env), lw=1.0, color='#b0b0b0',
                    label='envelope (peak at 0)')
        ax.plot(t_ms, edc, lw=1.8, color='k', label='decay curve')
        if ex['end'] < len(full):
            ax.axvline(1000 * ex['end'] / sr, color='#d00000', ls='-.', lw=1.2,
                       label='sample_max_sec')

        for name in VARIANTS:
            slope, intercept = ex['fits'][name]
            if not (np.isfinite(slope) and slope < 0):
                continue
            line = intercept + slope * np.arange(len(full)) / sr
            ax.plot(t_ms, line, lw=1.0, ls='--', color=COLOURS[name], alpha=0.9,
                    label=f'{name}  T60 {t60_from_slope(slope):.2f}s')
            window = WINDOWS.get(name)
            if window:
                inside = np.flatnonzero((edc <= window[0]) & (edc >= window[1]))
                if inside.size:
                    seg = slice(int(inside[0]), int(inside[-1]) + 1)
                    ax.plot(t_ms[seg], line[seg], lw=3.0, color=COLOURS[name])

        if np.isfinite(ex['t60_late']):
            ax.axhline(LATE_START_DB, color='#777777', ls=':', lw=1.0)
            ax.text(0.02, 0.04, f'late envelope T60 {ex["t60_late"]:.2f}s',
                    transform=ax.transAxes, fontsize=6.5, color='#444444')

        ax.set_ylim(max(-85.0, float(edc.min()) - 4), 4)
        ax.set_xlim(0, t_ms[-1] if len(full) else 1)
        ax.set_title(f'RT60 {ex["rt60"]:.2f}s · {ex["distance_m"]:.1f} m apart', fontsize=9)
        ax.set_xlabel('time (ms)', fontsize=8)
        ax.set_ylabel('dB', fontsize=8)
        ax.legend(fontsize=6, loc='upper right', framealpha=0.85)
        ax.grid(alpha=0.3)

    fig.suptitle('Which window describes the decay? Thick = fitted there, dashed = extrapolated',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_figure(frame, path, scene_dataset):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    pooled = np.concatenate([frame[f'drift_{n}'].dropna().values for n in VARIANTS])
    lo, hi = np.nanpercentile(pooled, [1, 99])
    for name in VARIANTS:
        s = frame[f'drift_{name}'].dropna()
        axes[0].hist(s, bins=np.linspace(lo, hi, 70), density=True, histtype='step',
                     lw=1.8, color=COLOURS[name], label=f'{name}  median {s.median():+.1f}')
        sub  = frame[['distance_m', f'drift_{name}']].dropna()
        bins = pd.cut(sub['distance_m'], np.arange(0, 21, 2))
        med  = sub.groupby(bins, observed=True)[f'drift_{name}'].median()
        axes[1].plot([b.mid for b in med.index], med.values, 'o-', lw=1.7,
                     color=COLOURS[name], label=name)
    axes[0].axvline(0, color='k', ls=':', lw=1.4)
    axes[0].set_xlabel('residual envelope slope (dB/s)   0 = perfectly flattened')
    axes[0].set_ylabel('density'); axes[0].legend(fontsize=8)
    axes[1].axhline(0, color='k', ls=':', lw=1.4)
    axes[1].set_xlabel('source-receiver distance (m)')
    axes[1].set_ylabel('median residual slope (dB/s)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle(f'{scene_dataset}: does scale_rir flatten the RIR?', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

SCALING_LEGEND = (
    ('fittable',  'share of RIRs the window produced a decay rate for'),
    ('drift',     'median residual envelope slope, dB/s; 0 perfect, below 0 leaves decay behind, '
                  'above 0 makes the signal grow'),
    ('|drift|',   'the same without sign, so under- and over-correction do not cancel'),
    ('over-tail', 'the 90th percentile of the signed drift: how far the over-correcting tenth '
                  'over-corrects. Scaling multiplies by exp(k*t), so a k that is too large grows '
                  'the quiet late samples without bound, while a k that is too small merely leaves '
                  'decay behind. The same error in k costs far more upward than downward, which is '
                  'why this column predicts a crashed run where |drift| does not.'),
    ('worst 10%', 'the 90th percentile of |drift|, which treats both directions alike'),
    ('over fp16', 'scaled RIRs exceeding 65,504, which turn the training loss to NaN'),
)

RT60_LEGEND = (
    ('T60 med',     'median estimated T60 in seconds, read as 60 / -slope'),
    ('bias %',      'median error against the reference, as a percentage of it; above 0 reads too '
                    'reverberant'),
    ('|err| %',     'the same without sign, so over- and under-estimates do not cancel'),
    ('worst 10%',   'the 90th percentile of |err| %, i.e. how wrong the bad tenth gets'),
    ('room spread', 'pooled within-room standard deviation of T60, in seconds. T60 belongs to the '
                    'room, so 0 would mean every pair measured in a room agrees'),
    ('dist r',      'correlation of T60 with source-receiver distance inside a room, after removing '
                    "each room's own mean. 0 is what physics expects, since moving the microphone "
                    'does not change the room'),
    ('rooms',       f'rooms holding at least {MIN_ROOM_PAIRS} measurements, which the last two '
                    f'columns are built from'),
)


def coverage_lines(frame):
    """How much decay the files hold, which is what limits every accuracy measure here."""
    lines = [f'Recorded length: median {frame["rec_len_s"].median():.3f} s.']
    for depth in DIAGNOSTIC_DB:
        name  = f'obs{int(-depth)}'
        share = frame[name] / frame['rec_len_s']
        lines.append(f'The {int(-depth)} dB crossing is reached on '
                     f'{100 * frame[name].notna().mean():.1f}% of RIRs, at a median '
                     f'{100 * share.median():.0f}% of the way through the file, and past 90% of it '
                     f'on {100 * (share > 0.9).mean():.0f}% of them.')
    lines.append('A crossing sitting at the end of the file is not a measurement of the decay: the '
                 'decay curve integrates the energy still to come, so it falls vertically wherever '
                 'the recording stops. That is why RT60 accuracy is scored against the envelope.')
    return lines


def scaling_table(frame):
    """Headings and formatted rows for the scaling comparison, shared by both renderers."""
    head = ['k from', 'fittable', 'drift', '|drift|', 'over-tail', 'worst 10%', 'over fp16']
    rows = []
    for name in VARIANTS:
        s = frame[f'drift_{name}'].dropna()
        if s.empty:
            continue
        rows.append([name,
                     f'{100 * frame[f"k_{name}"].notna().mean():.1f}%',
                     f'{s.median():+.1f}',
                     f'{s.abs().median():.1f}',
                     f'{s.quantile(.9):+.1f}',
                     f'{s.abs().quantile(.9):.1f}',
                     f'{int((frame[f"peak_{name}"] > FP16_MAX).sum()):,}'])
    return head, rows


def rt60_table(frame):
    """Headings and formatted rows for the RT60 comparison, shared by both renderers."""
    head = ['k from', 'T60 med', 'bias %', '|err| %', 'worst 10%', 'room spread', 'dist r', 'rooms']
    rows = []
    for name in VARIANTS:
        t60 = frame[f't60_{name}'].dropna()
        err = frame[f't60_err_pct_{name}'].dropna()
        if t60.empty:
            continue
        spread, coupling, n_rooms = within_room(frame, f't60_{name}')
        rows.append([name, f'{t60.median():.3f}', f'{err.median():+.1f}',
                     f'{err.abs().median():.1f}', f'{err.abs().quantile(.9):.1f}',
                     f'{spread:.3f}', f'{coupling:+.3f}', f'{n_rooms:,}'])
    return head, rows


def distance_table(frame):
    """Whether the leftover drift tracks how far apart the pair was."""
    head, rows = ['k from', 'drift vs distance r'], []
    for name in VARIANTS:
        sub = frame[['distance_m', f'drift_{name}']].dropna()
        if len(sub) > 10:
            r = np.corrcoef(sub['distance_m'], sub[f'drift_{name}'])[0, 1]
            rows.append([name, f'{r:+.3f}'])
    return head, rows


def reference_note(frame):
    late = frame['t60_late'].dropna()
    return (f'Every window is scored against the envelope slope over the late part of the signal, '
            f'from {LATE_START_DB:.0f} dB to the end of the recording, expressed as a T60: median '
            f'{late.median():.3f} s, available on {100 * frame["t60_late"].notna().mean():.1f}% of '
            f'RIRs. No window is fitted there, so all of them extrapolate onto it.')


def _print_table(head, rows):
    width = [max(len(head[i]), *(len(r[i]) for r in rows)) + 2 for i in range(len(head))]
    for line in (head, None, *rows):
        if line is None:
            print('-' * sum(width))
            continue
        print(''.join(c.ljust(width[i]) if i == 0 else c.rjust(width[i])
                      for i, c in enumerate(line)))


def _print_legend(legend):
    pad = max(len(term) for term, _ in legend) + 2
    for term, text in legend:
        body = textwrap.wrap(text, 96 - pad)
        print(f'  {term.ljust(pad)}{body[0]}')
        for extra in body[1:]:
            print(f'  {" " * pad}{extra}')


def report_to_stdout(frame):
    print('\nHOW MUCH DECAY THE RECORDINGS ACTUALLY HOLD\n')
    for line in coverage_lines(frame):
        print(textwrap.fill(line, 96, initial_indent='  ', subsequent_indent='  '))

    print('\n\nSCALING — how flat is the RIR after scale_rir multiplies by exp(k*t)?\n')
    _print_table(*scaling_table(frame))
    print()
    _print_legend(SCALING_LEGEND)
    print('\n  Does the leftover drift depend on source-receiver distance?\n')
    _print_table(*distance_table(frame))

    print('\n\nRT60 — how well does each window describe the decay it was not fitted to?\n')
    _print_table(*rt60_table(frame))
    print()
    print(textwrap.fill(reference_note(frame), 96, initial_indent='  ', subsequent_indent='  '))
    print()
    _print_legend(RT60_LEGEND)


def preserved_notes(path):
    """Keep anything written between the note markers across a regeneration."""
    start, end = NOTES_RE
    if path.exists():
        text = path.read_text()
        if start in text and end in text:
            return text.split(start)[1].split(end)[0]
    return '\n\n_Nothing yet. Anything written here survives regeneration._\n\n'


def write_report(path, frame, args, figure_dir, csv_path):
    """The readable companion to scaling_check.csv, which is built for processing not reading."""
    def row(cells):
        return '| ' + ' | '.join(c.replace('|', r'\|') for c in cells) + ' |'

    def table(builder):
        head, rows = builder(frame)
        return [row(head), '|' + '|'.join(['---'] * len(head)) + '|', *(row(r) for r in rows)]

    def legend(entries):
        return [f'- **`{term}`** — {text}' for term, text in entries]

    windows = ', '.join(f'`[{hi:.0f}, {lo:.0f}]`' for hi, lo in WINDOWS.values())
    lines = [
        f'# {args.scene_dataset} decay-window comparison', '',
        f'_Generated by `13_scaling_check.py` on '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}._',
        f'_Figures: `{figure_dir}` · per-RIR measurements: `{csv_path}`_', '',
        '## Notes', '', NOTES_RE[0], preserved_notes(path), NOTES_RE[1], '',
        '## Run', '',
        f'- **Windows:** {windows}',
        f'- **Population:** {args.population}'
        + (f', per-room cap seeded {args.cap_seed}' if args.population == 'capped' else ''),
        f'- **RIRs measured:** {len(frame):,}'
        + (f' drawn at random, seed {args.seed}' if args.n_rirs else ' (the whole population)'),
        f'- **Audio:** sample_max_sec {args.sample_max_sec} s'
        + (f', resampled to {args.sr_target} Hz' if args.sr_target else ', native sample rate'),
        f'- **Flattening judged from:** {START_DB:.0f} dB to the end of the recorded signal', '',
        '## How much decay the recordings hold', '',
        *(f'- {line}' for line in coverage_lines(frame)), '',
        '## Scaling', '',
        'How flat is the RIR after `scale_rir` multiplies by `exp(k*t)`?', '',
        *table(scaling_table), '',
        *legend(SCALING_LEGEND), '',
        'Does the leftover drift depend on source-receiver distance?', '',
        *table(distance_table), '',
        '## RT60', '',
        'How well does each window describe the decay it was not fitted to?', '',
        *table(rt60_table), '',
        reference_note(frame), '',
        *legend(RT60_LEGEND), '',
        '## Figures', '',
        '- `scaling_examples.png` — envelope before and after scaling; a flat line means the '
        'decay was removed',
        '- `scaling_examples_waveform.png` — the same samples as waveforms, where a swelling '
        'tail is what overflows fp16',
        '- `scaling_examples_edc.png` — the same samples as decay curves, against the curve a '
        'perfect result would give',
        '- `decay_fits.png` — each window\'s fitted line over the decay curve, thick where '
        'fitted and dashed where extrapolated',
        '- `window_edges.png` — where each window edge falls on the signal itself',
        '- `scaling_check.png` — drift distributions and drift against distance', '',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


# ---------------------------------------------------------------------------

def pick_examples(rows, n):
    """n row labels spread across the RT60 range, so figures show dry through reverberant."""
    order = rows['rt60'].rank(pct=True)
    return [rows.index[(order - q).abs().argmin()] for q in np.linspace(0.05, 0.95, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene-dataset', choices=sorted(DATASET_DEFAULTS), required=True)
    ap.add_argument('--n-rirs', type=int, default=20000,
                    help='How many RIRs to measure, drawn at random. 0 measures the whole population')
    ap.add_argument('--population', choices=('capped', 'all'), default='capped',
                    help="'capped' applies the per-room cap, giving the rows training sees; "
                         "'all' keeps every valid pair, which big rooms then dominate")
    ap.add_argument('--cap-seed', type=int, default=42,
                    help="Seed for the per-room cap; must match training's --random-seed "
                         'for the measured rows to be the rows training serves')
    ap.add_argument('--n-examples', type=int, default=10,
                    help='RIRs plotted before and after scaling, spread across the RT60 range')
    ap.add_argument('--n-fit-examples', type=int, default=12,
                    help='RIRs whose decay curve and fitted lines are plotted')
    ap.add_argument('--n-waveform-examples', type=int, default=10,
                    help='RIRs whose waveform is plotted with the window edges marked')
    ap.add_argument('--seed', type=int, default=42, help='Seed for the random draw')
    ap.add_argument('--sr-target', type=int, default=None, help='Resample before measuring')
    ap.add_argument('--sample-max-sec', type=float, default=0.6, help='Window training uses')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--report', default=None,
                    help='Where the written report goes (default: scaling_windows_report.md '
                         'in the output directory)')
    args = ap.parse_args()

    d       = DATASET_DEFAULTS[args.scene_dataset]
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(project_root) / 'data' / args.scene_dataset / 'analysis'
    (out_dir / 'figures').mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(d['mapping_csv'], comment='#')
    rows    = mapping[mapping['valid'] == True]
    if args.scene_dataset == 'matterport3d':
        rows = rows[rows['same_region'] == True]
    rows = rows[rows['rt60'].notna()]

    # The per-room cap is what decides which rows training ever sees, and it is the
    # dataset's own function rather than a copy of it, so the rows match row for row.
    if args.population == 'capped':
        cap  = DATASET_DEFAULTS[args.scene_dataset]['max_pairs_per_room']
        full = len(rows)
        if cap is not None:
            rows = _subsample_rooms(rows, _room_keys(rows, args.scene_dataset),
                                    cap, args.cap_seed)
        print(f'Per-room cap of {cap} (seed {args.cap_seed}) leaves {len(rows):,} '
              f'of {full:,} valid pairs — the population training serves')
    else:
        print(f'No per-room cap: all {len(rows):,} valid pairs, which the largest rooms dominate')

    if args.n_rirs and args.n_rirs < len(rows):
        rows = rows.sample(args.n_rirs, random_state=args.seed)
        print(f'Measuring {len(rows):,} RIRs drawn at random (seed {args.seed}) from {args.scene_dataset}')
    else:
        print(f'Measuring every one of the {len(rows):,} valid pairs in {args.scene_dataset}')

    env_labels  = pick_examples(rows, args.n_examples)
    fit_labels  = pick_examples(rows, args.n_fit_examples)
    wave_labels = pick_examples(rows, args.n_waveform_examples)
    wanted      = set(env_labels) | set(fit_labels) | set(wave_labels)
    kept, records, t0 = {}, [], time.time()

    for i, (label, row) in enumerate(zip(rows.index, rows.itertuples(index=False)), 1):
        full, served, sr, end = load_rir(d['rir_root'], row, args.sr_target, args.sample_max_sec)
        if full.size == 0 or not np.any(full):
            continue

        per_window, shared = measure(full, served, sr, end)
        rec = {'scene': row.scene, 'receiver_idx': row.receiver_idx,
               'source_idx': row.source_idx, 'room': room_label(row, args.scene_dataset),
               'distance_m': row.distance_m, 'rt60': row.rt60, **shared}
        for name, values in per_window.items():
            rec.update({f'{field}_{name}': value for field, value in values.items()})
        records.append(rec)

        if label in wanted:
            fits, edc_full = decay_fits(full, sr)
            after = np.flatnonzero(compute_edc(served) <= START_DB)
            kept[label] = {'full': full, 'rir': served, 'sr': sr, 'edc': edc_full,
                           'start': int(after[0]) if after.size else 0, 'end': end,
                           'fits': fits, 'rt60': row.rt60, 'distance_m': row.distance_m,
                           't60_late': shared['t60_late']}

        if i % max(1, len(rows) // 20) == 0:
            done = time.time() - t0
            print(f'  [{100*i/len(rows):5.1f}%] {i:,}/{len(rows):,}  '
                  f'elapsed {timedelta(seconds=int(done))}  '
                  f'ETA {timedelta(seconds=int(done/i*(len(rows)-i)))}', flush=True)

    frame = pd.DataFrame(records)
    print(f'\n{len(frame):,} RIRs measured')
    report_to_stdout(frame)

    summary_figure(frame, out_dir / 'figures' / 'scaling_check.png', args.scene_dataset)
    for labels, filename, plot in (
            (env_labels,  'scaling_examples.png',          examples_figure),
            (env_labels,  'scaling_examples_waveform.png', waveform_examples_figure),
            (env_labels,  'scaling_examples_edc.png',      edc_examples_figure),
            (fit_labels,  'decay_fits.png',                decay_fits_figure),
            (wave_labels, 'window_edges.png',              window_edges_figure)):
        chosen = [kept[label] for label in dict.fromkeys(labels) if label in kept]
        if chosen:
            chosen.sort(key=lambda e: e['rt60'])
            plot(chosen, out_dir / 'figures' / filename)

    csv_path = out_dir / 'scaling_check.csv'
    frame.to_csv(csv_path, index=False)
    report_path = Path(args.report) if args.report else out_dir / 'scaling_windows_report.md'
    write_report(report_path, frame, args, out_dir / 'figures', csv_path)

    print(f'\nFigures written to {out_dir / "figures"}')
    print(f'Per-RIR measurements written to {csv_path}')
    print(f'Report written to {report_path}')


if __name__ == '__main__':
    main()
