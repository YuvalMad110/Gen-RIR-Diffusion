"""
Synthetic RIR generation utilities.

Provides functions for generating Room Impulse Responses using:
- pyroomacoustics (image source method)
- Habets' RIR-Generator (image source method)

Used by synthetic_eval.py and full_model_eval.py (baseline).
"""

import numpy as np
import pyroomacoustics as pra
import rir_generator
from tqdm import tqdm


def generate_synthetic_rir_pra(room_dims, mic_pos, speaker_pos, rt60, sr, max_order=None):
    """Generate a synthetic RIR using pyroomacoustics.

    Args:
        room_dims: [length, width, height] in meters
        mic_pos: [x, y, z] microphone position in meters
        speaker_pos: [x, y, z] speaker position in meters
        rt60: Target RT60 in seconds
        sr: Sample rate
        max_order: Maximum reflection order (None = auto based on rt60)

    Returns:
        rir: RIR waveform as numpy array
    """
    e_absorption, max_order_calc = pra.inverse_sabine(rt60, room_dims)

    if max_order is None:
        max_order = max_order_calc

    room = pra.ShoeBox(
        room_dims,
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        ray_tracing=False,
        air_absorption=True
    )

    room.add_source(speaker_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()

    rir = room.rir[0][0]
    return rir


def generate_synthetic_rir_habets(room_dims, mic_pos, speaker_pos, rt60, sr, nsample=None):
    """Generate a synthetic RIR using Habets' RIR-Generator (image source method).

    Args:
        room_dims: [length, width, height] in meters
        mic_pos: [x, y, z] microphone position in meters
        speaker_pos: [x, y, z] speaker position in meters
        rt60: Target RT60 in seconds
        sr: Sample rate
        nsample: Number of output samples (None = auto)

    Returns:
        rir: RIR waveform as numpy array
    """
    kwargs = dict(
        c=340,
        fs=sr,
        r=[mic_pos],
        s=speaker_pos,
        L=room_dims,
        reverberation_time=rt60,
    )
    if nsample is not None:
        kwargs['nsample'] = nsample

    h = rir_generator.generate(**kwargs)

    # h shape is (nsample, n_receivers) — squeeze to 1D for single mic
    rir = h[:, 0]
    return rir


RIR_METHODS = {
    'pra': generate_synthetic_rir_pra,
    'habets': generate_synthetic_rir_habets,
}


def generate_synthetic_rirs_batch(conditions, sr, max_length_samples=None, method='pra', verbose=True, rt60_estimates=None, dataset_name='gtu'):
    """Generate synthetic RIRs for a batch of conditions.

    Args:
        conditions: np.array of shape [n_samples, 10] or [n_samples, 9], or list of condition arrays.
                   Format: [room_length, room_width, room_height, mic_x, mic_y, mic_z, speaker_x, speaker_y, speaker_z, (rt60)]
        sr: Sample rate
        max_length_samples: Maximum RIR length in samples (for truncation/padding)
        method: RIR generation method ('pra' for pyroomacoustics, 'habets' for RIR-Generator)
        verbose: Show tqdm progress bar (default True)
        rt60_estimates: Per-sample RT60 values (seconds). Required when conditions are 9-dim (no RT60 in vector).
        dataset_name: 'soundspaces' or 'gtu'. SoundSpaces positions are room-center-relative and must be
                      shifted by (L/2, W/2) before passing to the generators.

    Returns:
        rirs: List of RIR waveforms (numpy arrays, float32)
    """
    generate_fn = RIR_METHODS[method]
    method_label = 'pyroomacoustics' if method == 'pra' else 'RIR-Generator (Habets)'
    rirs = []

    iterator = range(len(conditions))
    if verbose:
        iterator = tqdm(iterator, desc=f"Generating synthetic RIRs ({method_label})")

    for i in iterator:
        cond = conditions[i]
        room_dims = cond[:3].tolist()
        mic_pos = cond[3:6].tolist()
        speaker_pos = cond[6:9].tolist()
        if len(cond) > 9:
            rt60 = float(cond[9])
        elif rt60_estimates is not None:
            rt60 = float(rt60_estimates[i])
        else:
            raise ValueError("RT60 not in condition vector and no rt60_estimates provided")

        # SoundSpaces u,v are room-center-relative; generators require corner-relative [0,L]x[0,W]
        if dataset_name == 'soundspaces':
            L, W = room_dims[0], room_dims[1]
            mic_pos     = [mic_pos[0]     + L / 2, mic_pos[1]     + W / 2, mic_pos[2]]
            speaker_pos = [speaker_pos[0] + L / 2, speaker_pos[1] + W / 2, speaker_pos[2]]

        extra_kwargs = {}
        if method == 'habets' and max_length_samples is not None:
            extra_kwargs['nsample'] = max_length_samples

        rir = generate_fn(room_dims, mic_pos, speaker_pos, rt60, sr, **extra_kwargs)

        # Handle length
        if max_length_samples is not None:
            if len(rir) > max_length_samples:
                rir = rir[:max_length_samples]
            elif len(rir) < max_length_samples:
                rir = np.pad(rir, (0, max_length_samples - len(rir)))

        rirs.append(rir.astype(np.float32))

    return rirs
