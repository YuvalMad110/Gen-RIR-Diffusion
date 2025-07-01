import os
import socket
import shutil
import matplotlib.pyplot as plt
import torch
import yaml
from datetime import datetime, timezone, timedelta
import pytz
import numpy as np
import re

def count_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

def get_project_root():
    """
    Returns the absolute path to the folder where this function is defined.
    This should be placed in a script located at the project root.
    """
    return os.path.abspath(os.path.dirname(__file__))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_timestamped_logdir(subdir_name="runs"):
    """Generate a full log_dir path in main script's directory with Israel timezone timestamp."""
    hostname = socket.gethostname()
    # Get timestamp in Israel timezone
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%b%d_%H-%M-%S")
    # Path to the main script (not where it's called from)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Combine path
    log_dir = os.path.join(base_path, subdir_name, f"{timestamp}_{hostname}")
    return log_dir


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def is_main_process(ddp_enabled):
    # Check if the current process is the main process (in case of DDP)
    return not ddp_enabled  or torch.distributed.get_rank() == 0

def save_metric_old(metric_per_epoch, metric_name, save_path, apply_log=False):
    """
    Saves a graph of a given metric (e.g., loss or accuracy) over training epochs.
    Optionally applies natural logarithm to the metric before plotting.

    Args:
        metric_per_epoch (list or array-like): Values of the metric for each epoch.
        metric_name (str): Name of the metric (used in title and filename).
        save_path (str): Directory path to save the plot. File will be named as '<metric_name>.png'.
        apply_log (bool): Whether to apply natural log to the metric values before plotting.
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Apply log if needed
    if apply_log:
        metric_vals = np.log(np.array(metric_per_epoch))
        name_suffix = f"log_{metric_name}"
        title = f'Log({metric_name}) per Epoch'
    else:
        metric_vals = metric_per_epoch
        name_suffix = metric_name
        title = f'{metric_name.capitalize()} per Epoch'

    # Create and save plot
    plot_filename = os.path.join(save_path, f"{name_suffix}.png")
    plt.figure()
    plt.plot(range(1, len(metric_vals) + 1), metric_vals, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(name_suffix)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

def save_metric(metric_per_epoch, metric_name, save_path, apply_log=False):
    """
    Saves a graph of training metrics over epochs.
    If input is a list, plots a single metric.
    If input is a dictionary, plots multiple metrics on the same graph.

    Args:
        metric_per_epoch (list or dict): List of metric values or a dict of lists keyed by metric names.
        metric_name (str): Base name for the plot and saved file.
        save_path (str): Directory to save the plot.
        apply_log (bool): Whether to apply natural log to the metric values before plotting.
    """
    os.makedirs(save_path, exist_ok=True)
    plt.figure()

    if isinstance(metric_per_epoch, dict):
        for key, values in metric_per_epoch.items():
            if apply_log:
                values = np.log(np.array(values))
            plt.plot(range(1, len(values) + 1), values, marker='o', label=key)
        plt.legend()
    else:
        values = np.log(np.clip(np.array(metric_per_epoch), a_min=1e-8, a_max=None)) if apply_log else metric_per_epoch
        plt.plot(range(1, len(values) + 1), values, marker='o')

    title = f"{'Log ' if apply_log else ''}{metric_name.capitalize()} per Epoch"
    ylabel = "Log Loss" if apply_log else "Loss"
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    
    full_path = os.path.join(save_path, f"{metric_name}.png")
    plt.savefig(full_path)
    plt.close()

def extract_losses_from_log(log_path):
    """
    Extracts loss values from a log file with lines containing 'Loss: <value>'.
    """
    with open(log_path, "r") as f:
        content = f.read()
    losses = [float(match) for match in re.findall(r"Loss:\s+([0-9.]+)", content)]
    return losses

def get_israel_time():
    tz = pytz.timezone("Asia/Jerusalem")
    return datetime.now(tz)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def spectrogram_to_waveform(spectrogram, n_fft=256, hop_length=128, length=None, device="cpu"):
    """
    Reconstruct waveform from spectrogram with shape [2, F, T] (real and imag parts).
    
    Args:
        spectrogram (Tensor): Tensor of shape [2, F, T]
        n_fft (int): FFT size used during STFT
        hop_length (int): Hop length used during STFT
        length (int, optional): Expected output length of the waveform
        device (str or torch.device): Device to use
    
    Returns:
        waveform (Tensor): Reconstructed waveform (1D tensor)
    """
    # Reconstruct complex STFT
    real, imag = spectrogram[0], spectrogram[1]
    complex_spec = torch.complex(real, imag)
    # Create Hann window
    window = torch.hann_window(n_fft, device=device)
    # Apply inverse STFT
    waveform = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length,
                           window=window, length=length)
    return waveform

def plot_signals(signals, legend=None, title="Signal Plot", save_path=None):
    """
    Plot multiple signals on a single plot.
    
    Args:
        signals: List of 1D arrays/signals or 2D matrix (each row is a signal)
        legend: List of legend labels. If None, uses indices (1, 2, ...)
        title: Plot title
        save_path: Path to save the plot. If None, only displays the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy array if not already
    if not isinstance(signals, np.ndarray):
        signals = np.array(signals)
    
    # Handle different input formats
    if signals.ndim == 1:
        # Single signal - convert to 2D
        signals = signals.reshape(1, -1)
    elif signals.ndim == 2:
        # Multiple signals - check if we need to transpose
        # Assume longer dimension is time, shorter is number of signals
        if signals.shape[0] > signals.shape[1]:
            signals = signals.T  # Transpose so each row is a signal
    
    n_signals = signals.shape[0]
    signal_length = signals.shape[1]
    
    # Create time axis
    time = np.arange(signal_length)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot each signal
    for i in range(n_signals):
        plt.plot(time, signals[i], linewidth=1.0, alpha=0.8)
    
    # Set up legend
    if legend is None:
        legend = [f'Signal {i+1}' for i in range(n_signals)]
    elif len(legend) != n_signals:
        print(f"Warning: Legend length ({len(legend)}) doesn't match number of signals ({n_signals})")
        legend = [f'Signal {i+1}' for i in range(n_signals)]
    
    plt.legend(legend)
    
    # Set labels and title
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show the plot (this will open in VS Code if running in VS Code)
    plt.show()