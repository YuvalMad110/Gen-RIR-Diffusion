"""
Data loading utilities for RIR inference.

Functions list:
- load_pretrained_model: Load model from a modern run directory (has run_config.json).
- data_params_from_run_config: Extract a data_info-compatible dict from run_config['args'].
- load_model_and_data_info: Legacy loader for old GTU checkpoints (no run_config.json).
- build_test_dataloader: Reconstruct the test-split DataLoader from run_config or data_info.
- build_condition_tensor: Build the scalar conditioning tensor from a batch dict.
- load_dataset_conditions: Load conditions and RIRs from dataset (run_inference.py helper).
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

    # Instantiate ImageEncoder only when image conditioning was active during training
    # (image_root=None means the encoder was never used even if image_encoder_config is present)
    image_encoder = None
    ie_config = run_config.get('image_encoder_config')
    if ie_config is not None and run_config['args'].get('image_root') is not None:
        from image_encoder import ImageEncoder
        cross_attention_dim = model_config['block_out_channels'][-1]
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
        'split_by_room':       a.get('split_by_room', False),
        'random_seed':         a.get('random_seed', 42),
        'dataset_name':        a.get('dataset_name', 'gtu'),
        'use_rt60_condition':  a.get('use_rt60_condition', False),
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
    """Build a test-split DataLoader from data_params (and optionally run_config for SoundSpaces).

    For GTU: collate_fn=None, yields 5-tuples (rir, room_dim, mic_loc, speaker_loc, rt60).
    For SoundSpaces: scale_and_spectrogram_collate_fn with scale_rir_flag=False,
        yields dicts with keys 'rir', 'room_dim', 'mic_loc', 'speaker_loc', and optionally 'images'.

    Args:
        data_params:  Dict as returned by data_params_from_run_config or legacy data_info.
        batch_size:   Batch size for the DataLoader.
        workers:      Number of DataLoader workers.
        run_config:   Full run_config dict (needed for SoundSpaces dataset params).

    Returns:
        (test_dataset, test_dataloader)
    """
    import torch.utils.data
    from data.rir_dataset import load_rir_dataset

    dataset_name = data_params.get('dataset_name', 'gtu')
    nSamples = data_params.get('nSamples')

    if dataset_name == 'gtu':
        args = run_config['args'] if run_config else {}
        _, _, test_dataset = load_rir_dataset(
            name='gtu',
            path=args.get('data_path', './datasets/GTU_rir/GTU_RIR.pickle.dat'),
            split=True, mode='raw',
            hop_length=data_params['hop_length'], n_fft=data_params['n_fft'],
            use_spectrogram=True,
            sample_max_sec=data_params['sample_max_sec'],
            nSamples=nSamples,
            sr_target=data_params['sr_target'],
            train_ratio=data_params['train_ratio'],
            eval_ratio=data_params['eval_ratio'],
            test_ratio=data_params['test_ratio'],
            random_seed=data_params['random_seed'],
            split_by_room=data_params['split_by_room'],
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, collate_fn=None, drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    else:  # soundspaces
        if run_config is None:
            raise ValueError("run_config is required for SoundSpaces test dataloader")
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
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, collate_fn=collate_fn, drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    return test_dataset, test_dataloader


def build_condition_tensor(batch, device: torch.device) -> torch.Tensor:
    """Build the scalar conditioning tensor from a dataloader batch.

    Handles both dict batches (SoundSpaces collate_fn) and tuple batches (GTU default collate).

    Returns:
        Tensor of shape [B, 9] for SoundSpaces (no RT60) or [B, 10] for GTU (with RT60).
    """
    if isinstance(batch, dict):
        parts = [batch['room_dim'], batch['mic_loc'], batch['speaker_loc']]
        if 'rt60' in batch:
            parts.append(batch['rt60'].unsqueeze(1))
        return torch.cat(parts, dim=1).float().to(device)
    else:
        # GTU tuple: (rirs, room_dims, mic_locs, speaker_locs, rt60s)
        _, room_dims, mic_locs, speaker_locs, rt60s = batch
        return torch.cat([room_dims, mic_locs, speaker_locs, rt60s.unsqueeze(1)], dim=1).float().to(device)


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


# =============================================================================
# run_inference.py helper
# =============================================================================

def load_dataset_conditions(dataset, n_samples: int, data_info: Dict,
                            rir_indices: Optional[List[int]] = None
                            ) -> Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray], List[int], Optional[torch.Tensor]]:
    """Load conditions and RIRs from dataset."""
    # Select indices
    if rir_indices is None:
        rir_indices = np.random.choice(len(dataset), size=n_samples, replace=False).tolist()
    else:
        if len(rir_indices) != n_samples:
            print(f"Warning: Using first {n_samples} of {len(rir_indices)} provided indices")
            rir_indices = rir_indices[:n_samples]

    print(f"Loading dataset samples at indices: {rir_indices}")

    hop_length = data_info['hop_length']
    n_fft = data_info['n_fft']

    conditions = []
    real_rirs_wave = []
    real_rirs_spec = []

    for idx in rir_indices:
        rir, room_dim, mic_loc, speaker_loc, rt60 = dataset[idx]

        rir = _to_numpy(rir)
        room_dim = _to_numpy(room_dim)
        mic_loc = _to_numpy(mic_loc)
        speaker_loc = _to_numpy(speaker_loc)
        rt60 = _to_numpy(rt60)

        if rir.ndim == 3 and rir.shape[0] == 2:
            rir_spec = rir
            rir_wave = spectrogram_to_waveform(rir, hop_length, n_fft)
        else:
            rir_wave = rir.squeeze()
            rir_spec = waveform_to_spectrogram(rir_wave, hop_length, n_fft)

        real_rirs_wave.append(rir_wave)
        real_rirs_spec.append(rir_spec)
        condition = np.concatenate([room_dim, mic_loc, speaker_loc, [rt60]])
        conditions.append(condition)

    conditions_tensor = torch.tensor(conditions, dtype=torch.float32)

    k_factors = None
    if data_info.get('scale_rir', False):
        real_tensor = torch.stack([torch.tensor(r, dtype=torch.float32) for r in real_rirs_wave])
        edc = calculate_edc(real_tensor)
        k_factors, _ = estimate_decay_k_factor(edc, data_info['sr_target'], data_info['db_cutoff'])
        print(f"Calculated k-factors: {k_factors.tolist()}")

    return conditions_tensor, real_rirs_wave, real_rirs_spec, rir_indices, k_factors


def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.asarray(x)
