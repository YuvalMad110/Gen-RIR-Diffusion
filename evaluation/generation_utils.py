"""
Everything about where a model's generated RIRs are kept: how the place is named, how a set is
written and read back, and when one already there can be reused. Nothing else belongs here.

pre_generate.py writes through this module and run_evaluation.py reads through it, so the two
scripts never import each other and cannot drift apart on the naming or the file format.

A set lives at <run_dir>/generations/<tag>/ holding generations.npz and generation_config.json.
The tag is derived from the sampler settings rather than typed, so the same settings always land
in the same place, and settings that would produce different waveforms cannot quietly overwrite
each other.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

STORE_NAME  = 'generations.npz'
CONFIG_NAME = 'generation_config.json'

# The settings that decide the waveforms. A set already on disk is reused only when all of these
# agree, so anything that would generate differently is caught instead of silently reused.
SETTINGS_KEYS = ('split_id', 'sampler', 'num_inference_steps', 'guidance_scale', 'seed',
                 'batch_size', 'limit', 'unscaling', 'sr_target', 'checkpoint_size_bytes',
                 'checkpoint_modified')


# =============================================================================
# Naming
# =============================================================================

def effective_sampler(use_ddim: bool, num_inference_steps: int, n_timesteps: int) -> str:
    """
    The sampler that will actually run.

    Asking for fewer steps than the model was trained with forces DDIM inside
    RIRDiffusionModel.generate, so the name has to be derived rather than taken from the flag.

    Returns:
        'ddim' or 'ddpm'.
    """
    return 'ddim' if (use_ddim or num_inference_steps < n_timesteps) else 'ddpm'


def generation_tag(sampler: str, num_inference_steps: int, guidance_scale: float,
                   seed: int, limit=None) -> str:
    """
    Directory name for one set of sampler settings, e.g. 'ddim50_g2_s42'.

    A limited run carries '_n<limit>', so a partial set cannot be mistaken for a full one.
    """
    tag = f'{sampler}{num_inference_steps}_g{guidance_scale:g}_s{seed}'
    return f'{tag}_n{limit}' if limit else tag


def generation_dir(run_dir, tag: str) -> Path:
    """Where a generation set lives inside a run directory."""
    return Path(run_dir) / 'generations' / tag


# =============================================================================
# Reusing what is already there
# =============================================================================

def inspect_existing(target_dir: Path, settings: dict) -> tuple:
    """
    Whether a set already on disk can be reused.

    Args:
        target_dir: the set's directory
        settings:   the settings this run would generate with, keyed by SETTINGS_KEYS

    Returns:
        (state, detail) where state is one of:
            'absent'   nothing there, or an empty directory
            'match'    a complete set generated with these settings
            'differs'  a set generated with different settings; detail names them
            'partial'  a directory holding something other than a complete set
    """
    if not target_dir.exists() or not any(target_dir.iterdir()):
        return 'absent', ''

    store, config = target_dir / STORE_NAME, target_dir / CONFIG_NAME
    if not store.exists() or not config.exists():
        return 'partial', f'{target_dir} exists but holds no complete set'

    with open(config) as f:
        found = json.load(f)
    differing = [f'{k}: {found.get(k)!r} on disk against {settings.get(k)!r} now'
                 for k in SETTINGS_KEYS if found.get(k) != settings.get(k)]
    if differing:
        return 'differs', '; '.join(differing)
    return 'match', f"{found.get('n_samples', '?')} samples generated {found.get('created', '?')}"


# =============================================================================
# Store I/O
# =============================================================================

def save_generations(target_dir: Path, sample_ids, generated, config: dict) -> Path:
    """
    Write a generation set.

    Everything is written to a staging directory and moved into place only once complete, so a
    crash can leave no set rather than a partial one that would load and score.

    Args:
        target_dir: the set's directory
        sample_ids: [N] ids, in generation order
        generated:  [N, T] waveforms at the model's output length
        config:     the settings and provenance to record beside them

    Returns:
        Path to the .npz.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target_dir.parent, prefix='.staging-'))
    try:
        np.savez(staging / STORE_NAME, sample_id=np.asarray(sample_ids),
                 generated=np.asarray(generated, dtype=np.float32))
        with open(staging / CONFIG_NAME, 'w') as f:
            json.dump(config, f, indent=2)
        shutil.rmtree(target_dir, ignore_errors=True)
        os.replace(staging, target_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target_dir / STORE_NAME


def load_generations(target_dir) -> dict:
    """
    Read a generation set back.

    Returns:
        {'sample_id': [N], 'generated': [N, T], 'config': dict}
    """
    target_dir = Path(target_dir)
    store, config = target_dir / STORE_NAME, target_dir / CONFIG_NAME
    if not store.exists():
        raise FileNotFoundError(f'no generation set at {target_dir}')

    with np.load(store) as data:
        out = {'sample_id': data['sample_id'], 'generated': data['generated']}
    with open(config) as f:
        out['config'] = json.load(f)
    return out
