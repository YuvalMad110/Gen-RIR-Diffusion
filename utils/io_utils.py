"""
File I/O utilities for RIR inference.
"""

import os
import json
import hashlib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def save_inference_results(generated_rirs: List[np.ndarray], generated_specs: List[np.ndarray],
                           conditions: np.ndarray, config: Dict, model_path: str,
                           rir_indices: List[int], save_path: str, guidance_scale: float = 1.0):
    """Save inference results to file."""
    save_dict = {
        "generated_rirs": generated_rirs,
        "generated_specs": generated_specs,
        "conditions": conditions,
        "config": config,
        "model_path": model_path,
        "rir_indices": rir_indices,
        "guidance_scale": guidance_scale,
        "timestamp": datetime.now().isoformat()
    }
    
    output_path = Path(save_path) / "inference_results.pt"
    torch.save(save_dict, output_path)
    print(f"Inference results saved to: {output_path}")


def save_metrics(metrics: Dict, rir_indices: List[int], save_path: str,
                 conditions: Optional[np.ndarray] = None, guidance_scale: float = 1.0):
    """Save evaluation metrics to text file."""
    output_path = Path(save_path) / "metrics.txt"
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RIR Generation Metrics\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"RIR Indices: {rir_indices}\n\n")
        
        if conditions is not None:
            f.write("Conditions:\n")
            for i, idx in enumerate(rir_indices):
                cond = conditions[i]
                f.write(f"  [{idx}] Room: {cond[0]:.1f}x{cond[1]:.1f}x{cond[2]:.1f}m, RT60: {cond[-1]:.2f}s\n")
            f.write("\n")
        
        f.write("Metrics:\n")
        _write_metrics_recursive(f, metrics, indent=2)
    
    print(f"Metrics saved to: {output_path}")


def _write_metrics_recursive(f, metrics: Dict, indent: int = 0):
    """Recursively write metrics dictionary to file."""
    prefix = " " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            f.write(f"{prefix}{key}:\n")
            _write_metrics_recursive(f, value, indent + 2)
        elif isinstance(value, (list, np.ndarray)):
            if len(value) > 10:
                f.write(f"{prefix}{key}: [array of {len(value)} items]\n")
                f.write(f"{prefix}  mean: {np.mean(value):.4f}, std: {np.std(value):.4f}\n")
            else:
                f.write(f"{prefix}{key}: {value}\n")
        elif isinstance(value, float):
            f.write(f"{prefix}{key}: {value:.4f}\n")
        else:
            f.write(f"{prefix}{key}: {value}\n")


def save_config_summary(config: Dict, data_params: Dict, args: Any, save_path: str):
    """Save configuration summary for reproducibility."""
    summary = {
        "model_config": {k: v for k, v in config.items() if k != 'losses_per_epoch'},
        "data_params": data_params,
        "inference_args": {
            "model_path": args.model_path,
            "n_rirs": args.nRIR,
            "guidance_scale": getattr(args, 'guidance_scale', 1.0),
            "num_inference_steps": getattr(args, 'num_inference_steps', None),
            "seed": args.seed,
        },
        "timestamp": datetime.now().isoformat()
    }
    
    output_path = Path(save_path) / "config_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Config summary saved to: {output_path}")


_SPLIT_ID_COLUMNS = ['scene', 'receiver_region', 'receiver_idx', 'source_idx']


def save_split_identities(run_dir: str, datasets, names=('train', 'eval', 'test')) -> Path:
    """
    Write the rows each split holds, so the data a run saw stays recoverable even after
    the mapping CSV changes. Identity is the columns that name a pair, not row order.
    """
    frames = []
    for name, ds in zip(names, datasets):
        cols = [c for c in _SPLIT_ID_COLUMNS if c in ds.mapping.columns]
        frame = ds.mapping[cols].copy()
        frame.insert(0, 'split', name)
        frames.append(frame)

    out_path = Path(run_dir) / 'splits.csv.gz'
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False, compression='gzip')
    print(f"Split identities saved to: {out_path}  ({sum(len(f) for f in frames):,} rows)")
    return out_path


def _fingerprint(path: Optional[str], started_at: datetime) -> Optional[Dict]:
    """Identify an input file, and flag it when it was written after the run began."""
    if not path or not os.path.exists(path):
        return None
    digest, lines = hashlib.md5(), 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            digest.update(chunk)
            lines += chunk.count(b'\n')
    stat = os.stat(path)
    return {
        'path':        str(path),
        'md5':         digest.hexdigest(),
        'size_bytes':  stat.st_size,
        'lines':       lines,
        'modified':    datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds'),
        'modified_during_run': stat.st_mtime > started_at.timestamp(),
    }


def save_run_config(run_dir: str, args, model_config: Dict, ie_config: Optional[Dict],
                    git_hash: str, dataset=None, started_at: Optional[datetime] = None) -> None:
    """Save all training settings as a single human-readable JSON for reproducibility."""
    import socket

    def _to_json(obj):
        if hasattr(obj, 'tolist'):   # torch.Size, tensors, numpy arrays
            return obj.tolist()
        return str(obj)

    saved_args = vars(args).copy()
    if dataset is not None:   # the rate the RIRs are stored at, and sr_target as resolved
        saved_args.update(sr_native=dataset.sr_native, sr_target=dataset.sr_target)

    record = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "git_hash": git_hash,
        "host": socket.gethostname(),
        "args": saved_args,
        "model_config": model_config,
        "image_encoder_config": ie_config,
    }

    if started_at is not None:
        record["inputs"] = {
            "started_at":   started_at.isoformat(timespec='seconds'),
            "mapping_csv":  _fingerprint(saved_args.get('mapping_csv'), started_at),
            "geometry_csv": _fingerprint(saved_args.get('geometry_csv'), started_at),
        }

    out_path = Path(run_dir) / "run_config.json"
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2, default=_to_json)


def create_output_directory(base_path: Optional[str], model_path: str,
                            create_timestamp: bool = True) -> Path:
    """Create output directory for inference results."""
    if base_path is None:
        base_path = os.path.join(os.path.dirname(model_path), "generated_rirs")
    
    output_path = Path(base_path)
    
    if create_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / timestamp
    
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
