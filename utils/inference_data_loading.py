"""
Data loading utilities for RIR inference.

Functions:
- load_pretrained_model: Load model from a run directory (has run_config.json).
- data_params_from_run_config: Extract a data_params dict from run_config['args'].
- build_test_dataloader: Build the test-split DataLoader from run_config.
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.signal_proc import waveform_to_spectrogram, spectrogram_to_waveform, calculate_edc, estimate_decay_k_factor


# =============================================================================
# Modern loader — for runs that have run_config.json
# =============================================================================

def load_pretrained_model(run_dir: str, device: torch.device):
    """Load model from a run directory that contains run_config.json.

    Reads image_encoder_config from run_config.json, instantiates ImageEncoder if
    needed, then reconstructs the model via RIRDiffusionModel.init_from_config.

    Returns:
        model:      RIRDiffusionModel ready for inference
        run_config: Full run_config dict (run_config['args'] holds all data/audio params)
    """
    from RIRDiffusionModel import RIRDiffusionModel

    run_dir = Path(run_dir)
    config_path = run_dir / 'run_config.json'
    ckpt_path = run_dir / 'model_best.pth.tar'

    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {run_dir}. "
                                f"Use load_model_and_data_info for legacy checkpoints.")

    with open(config_path) as f:
        run_config = json.load(f)

    print(f"Loading model from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_config = run_config['model_config']
    model_config.setdefault('pos_encoder_cfg', None)  # old checkpoints: no positional encoder

    # Instantiate ImageEncoder only when image conditioning was active during training
    # (image_root=None means the encoder was never used even if image_encoder_config is present)
    image_encoder = None
    ie_config = run_config.get('image_encoder_config')
    if ie_config is not None and run_config['args'].get('image_root') is not None:
        from image_encoder import ImageEncoder
        cross_attention_dim = model_config['encoder_hidden_dims'][-1]
        image_encoder = ImageEncoder(out_dim=cross_attention_dim, **ie_config)

    model = RIRDiffusionModel.init_from_config(model_config, device, image_encoder)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"Model loaded on {device}")
    print(f"  Sample size: {model_config['sample_size']}, Timesteps: {model_config['n_timesteps']}")
    if hasattr(model, 'guidance_enabled'):
        print(f"  Guidance: enabled={model.guidance_enabled}, dropout={model.guidance_dropout_prob}")

    return model, run_config


def data_params_from_run_config(run_config: Dict) -> Dict:
    """Extract a data_info-compatible parameter dict from run_config['args'].

    Provides the same keys that legacy data_info dicts contained so that
    evaluation code works uniformly regardless of whether it loaded via
    load_pretrained_model or load_model_and_data_info.
    """
    a = run_config['args']
    return {
        'n_fft':               a['n_fft'],
        'hop_length':          a['hop_length'],
        'use_spectrogram':     True,
        'sr_target':           a['sr_target'],
        'sample_max_sec':      a['sample_max_sec'],
        'nSamples':            a.get('nSamples'),
        'db_cutoff':           a.get('db_cutoff', -40.0),
        'scale_rir':           a.get('scale_rir', False),
        'apply_zero_tail':     a.get('apply_zero_tail', False),
        'train_ratio':         a.get('train_ratio', 0.7),
        'eval_ratio':          a.get('eval_ratio', 0.15),
        'test_ratio':          a.get('test_ratio', 0.15),
        'split_by_room':       a.get('split_by_room', True),
        'random_seed':         a.get('random_seed', 42),
        'dataset_name':        a['dataset_name'],
        'use_rt60_condition':  a['use_rt60_condition'],
    }


# =============================================================================
# Legacy loader — for old GTU checkpoints (no run_config.json)
# =============================================================================

def load_model_and_data_info(model_path: str, device: torch.device, model_class) -> Tuple[Any, Dict]:
    """Load trained diffusion model and data_info from a legacy checkpoint.

    For modern runs use load_pretrained_model instead — it handles image-conditioned
    models correctly.
    """
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_config = checkpoint['model_config']
    data_info = checkpoint.get('data_info', {})

    model = model_class.init_from_config(model_config, device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"Model loaded on {device}")
    print(f"  Sample size: {model_config['sample_size']}, Timesteps: {model_config['n_timesteps']}")
    if hasattr(model, 'guidance_enabled'):
        print(f"  Guidance: enabled={model.guidance_enabled}, dropout={model.guidance_dropout_prob}")

    return model, data_info


# =============================================================================
# Shared helpers — used by all eval scripts
# =============================================================================

def build_test_dataloader(data_params: Dict, batch_size: int, workers: int,
                          run_config: Optional[Dict] = None):
    """Build a test-split DataLoader from data_params and run_config.

    Yields dicts with keys 'rir', 'room_dim', 'mic_loc', 'speaker_loc', 'scene',
    and optionally 'rt60', 'images'.

    Args:
        data_params:  Dict as returned by data_params_from_run_config.
        batch_size:   Batch size for the DataLoader.
        workers:      Number of DataLoader workers.
        run_config:   Full run_config dict.

    Returns:
        (test_dataset, test_dataloader)
    """
    import torch.utils.data
    from data.rir_dataset import load_rir_dataset

    if run_config is None:
        raise ValueError("run_config is required for build_test_dataloader")

    dataset_name = data_params.get('dataset_name', 'soundspaces')
    nSamples = data_params.get('nSamples')

    if dataset_name == 'soundspaces':
        a = run_config['args']
        ie_config = run_config.get('image_encoder_config')

        rir_view_type = None if a.get('rir_view_type') == 'none' else a.get('rir_view_type')
        room_overview_type = None if a.get('room_overview_type') == 'none' else a.get('room_overview_type')
        image_size = tuple(ie_config['image_size']) if ie_config and 'image_size' in ie_config else None

        _, _, test_dataset = load_rir_dataset(
            name='soundspaces',
            rir_root=a.get('data_path'),
            image_root=a.get('image_root'),
            scenes=a.get('scenes'),
            split=True,
            split_by_room=data_params['split_by_room'],
            train_ratio=data_params['train_ratio'],
            eval_ratio=data_params['eval_ratio'],
            test_ratio=data_params['test_ratio'],
            random_seed=data_params['random_seed'],
            sample_max_sec=data_params['sample_max_sec'],
            sr_target=data_params['sr_target'],
            rir_view_type=rir_view_type,
            room_overview_type=room_overview_type,
            room_overview_config=a.get('room_overview_config'),
            image_size=image_size,
            use_rt60=a.get('use_rt60_condition', False),
        )

        from data.dataset_collate_fn import scale_and_spectrogram_collate_fn
        collate_fn = scale_and_spectrogram_collate_fn(
            sr=data_params['sr_target'],
            db_cutoff=data_params['db_cutoff'],
            n_fft=data_params['n_fft'],
            hop_length=data_params['hop_length'],
            scale_rir_flag=False,
            use_spectrogram=False,
            apply_zero_tail=False,
            dataset_type='soundspaces',
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name!r}. Expected 'soundspaces'.")

    if nSamples is not None:
        test_size = int(nSamples * data_params['test_ratio'])
        if test_size < len(test_dataset):
            test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=collate_fn, drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    return test_dataset, test_dataloader


def _normalize_batch_to_dict(batch) -> Dict:
    """Normalize a GTU 5-tuple batch into the same dict format SoundSpaces uses."""
    if isinstance(batch, dict):
        return batch
    rirs, room_dims, mic_locs, speaker_locs, rt60s = batch
    return {
        'rir': rirs,
        'room_dim': room_dims,
        'mic_loc': mic_locs,
        'speaker_loc': speaker_locs,
        'rt60': rt60s,
    }


