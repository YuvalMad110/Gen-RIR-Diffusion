"""
File I/O utilities for RIR inference.
"""

import os
import json
import torch
import numpy as np
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
