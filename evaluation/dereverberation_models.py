"""
dereverberation_models.py
=========================
Model library imported by dereverb_eval.py.

Relationship to dereverb_eval.py
---------------------------------
dereverb_eval.py  — the evaluation harness / script you run. It convolves speech
                    with real and generated RIRs, runs a dereverberation model on
                    the output, and measures how much the model degrades on generated
                    vs real RIRs.
dereverberation_models.py  — this file. Defines the DereverbModel base class and
                    concrete implementations. dereverb_eval.py calls
                    build_dereverb_model() to get a model instance; it never
                    instantiates model classes directly.

Models
------
unet      — Scene-agnostic U-Net trained on GTU/LibriSpeech.
            Requires the scene_agnostic_dereverberation project on disk.
nara_wpe  — Blind WPE dereverberation (no training required).
            Install: python3 -m pip install nara_wpe

Adding a new model
------------------
1. Subclass DereverbModel and implement dereverberate().
2. Register it in build_dereverb_model().
"""

from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import sys

import numpy as np
import scipy.io
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared constant (used by both models and dereverb_eval.py)
# ─────────────────────────────────────────────────────────────────────────────

DEREVERB_SR = 16_000   # working sample rate for all dereverberation models


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class DereverbModel(ABC):
    """Common interface for all dereverberation models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in file names and logs."""

    @property
    def min_speech_samples(self) -> int:
        """Minimum waveform length (samples) the model requires. 0 = no constraint."""
        return 0

    @abstractmethod
    def dereverberate(self, reverb_wav: np.ndarray) -> np.ndarray:
        """
        Dereverberate a single-channel 16-kHz waveform.

        Parameters
        ----------
        reverb_wav : float32 ndarray, shape (N,)

        Returns
        -------
        float32 ndarray, shape (N,)
        """


# ─────────────────────────────────────────────────────────────────────────────
# U-Net model  (scene_agnostic_dereverberation)
# ─────────────────────────────────────────────────────────────────────────────

_DEREVERB_ROOT = Path(__file__).parents[2] / "scene_agnostic_dereverberation"

# U-Net training constants
_UNET_K      = 512     # STFT window size
_UNET_OVL    = 0.75    # STFT overlap fraction
_UNET_FRAMES = 256     # temporal segment length fed to the model
_UNET_EPS    = 2.2204 * np.exp(-16)


def _load_unet_dependencies():
    """
    Lazily import scene_agnostic_dereverberation utilities.
    Adds the project root to sys.path so imports resolve correctly.
    """
    if str(_DEREVERB_ROOT) not in sys.path:
        sys.path.append(str(_DEREVERB_ROOT))

    import networks.model as dereverb_arch
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "dereverb_data_utils", str(_DEREVERB_ROOT / "data" / "utils.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    synt_win_mat = scipy.io.loadmat(str(_DEREVERB_ROOT / "synt_win.mat"))
    synt_win     = synt_win_mat["synt_win"]

    return dereverb_arch, _mod, synt_win


def _remove_module_str(state_dict: dict) -> dict:
    """Strip 'module.' prefix added by DataParallel."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


class UNetDereverbModel(DereverbModel):
    """
    Scene-agnostic U-Net dereverberation model.

    Requires the scene_agnostic_dereverberation project to be present on disk
    at ../scene_agnostic_dereverberation relative to the Gen-RIR-Diffusion root.
    """

    def __init__(self, model_dir: str, device: torch.device):
        self._device   = device
        self._arch, self._utils, self._synt_win = _load_unet_dependencies()
        self._net, self._stats = self._load(model_dir)

    @property
    def name(self) -> str:
        return "unet"

    @property
    def min_speech_samples(self) -> int:
        return int(_UNET_FRAMES * _UNET_K * (1 - _UNET_OVL) + _UNET_K)

    def _load(self, model_dir: str):
        model_dir = Path(model_dir)
        print(f"  Loading U-Net dereverberation model from:\n    {model_dir}")

        settings = {}
        with open(model_dir / "settings.txt") as f:
            for line in f:
                if ": " in line:
                    k, v = line.strip().split(": ", 1)
                    settings[k.strip()] = v.strip()

        ngf = int(settings.get("ngf", 64))

        stats_path = Path(settings["global_stats_file"])
        if not stats_path.is_absolute():
            stats_path = Path.home() / str(stats_path).lstrip("./")
        with open(stats_path, "rb") as f:
            log_max_clean, log_min_clean, log_max_reverb, log_min_reverb = pickle.load(f)

        net = self._arch.UNet(ngf=ngf, nc=1, kernel_size=4).to(self._device)

        checkpoints = sorted((model_dir / "checkpoints").glob("epoch*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {model_dir / 'checkpoints'}")
        ckpt_path = checkpoints[-1]

        state_dict = torch.load(ckpt_path, map_location=self._device)
        state_dict = _remove_module_str(state_dict)
        net.load_state_dict(state_dict)
        net.eval()
        print(f"  Checkpoint : {ckpt_path.name}  (ngf={ngf})")

        return net, (log_max_clean, log_min_clean, log_max_reverb, log_min_reverb)

    def dereverberate(self, reverb_wav: np.ndarray) -> np.ndarray:
        log_max_clean, log_min_clean, log_max_reverb, log_min_reverb = self._stats
        stft               = self._utils.stft
        istft              = self._utils.istft
        normalize_log_spec   = self._utils.normalize_log_spec
        denormalize_log_spec = self._utils.denormalize_log_spec

        reverb_wav = reverb_wav / 1.1 / (np.max(np.abs(reverb_wav)) + 1e-8)

        Z_mag, Z_phase = stft(reverb_wav, _UNET_K, _UNET_OVL)
        if Z_mag.shape[1] < _UNET_FRAMES:
            raise RuntimeError(
                f"Signal too short: {Z_mag.shape[1]} STFT frames, need >= {_UNET_FRAMES}"
            )

        Z_log  = np.log(Z_mag.T + _UNET_EPS)[np.newaxis, ...]
        Z_norm = normalize_log_spec(Z_log, log_max_reverb, log_min_reverb)
        Z_t    = torch.from_numpy(Z_norm.astype("float32"))

        time_len   = Z_t.shape[1]
        edge_index = time_len // _UNET_FRAMES
        residue    = time_len %  _UNET_FRAMES

        to_model = Z_t[:, : edge_index * _UNET_FRAMES, :-1].view(
            -1, 1, _UNET_FRAMES, _UNET_K // 2
        )
        if residue:
            last_part = Z_t[:, -_UNET_FRAMES:, :-1].view(-1, 1, _UNET_FRAMES, _UNET_K // 2)
            to_model  = torch.cat([to_model, last_part], dim=0)

        with torch.no_grad():
            pred = self._net(to_model.to(self._device))
        pred = pred.squeeze().cpu()

        if residue == 0:
            recon = pred.view(edge_index * _UNET_FRAMES, _UNET_K // 2)
        else:
            recon1 = pred[:-1].view(edge_index * _UNET_FRAMES, _UNET_K // 2)
            recon  = torch.cat((recon1, pred[-1, -residue:, :]), dim=0)

        recon = torch.cat((recon, Z_t[0, :, -1:]), dim=1)
        recon = denormalize_log_spec(recon, log_max_clean, log_min_clean)
        s_hat = istft(recon.numpy().T, Z_phase, self._synt_win)
        s_hat = s_hat / 1.1 / (np.max(np.abs(s_hat)) + 1e-8)
        return s_hat[: len(reverb_wav)]


# ─────────────────────────────────────────────────────────────────────────────
# nara_wpe model
# ─────────────────────────────────────────────────────────────────────────────

class NaraWPEModel(DereverbModel):
    """
    Blind dereverberation via Weighted Prediction Error (WPE).

    No training or model files required.
    Install: python3 -m pip install nara_wpe

    Parameters
    ----------
    taps       : WPE filter length. Higher values capture more late reverberation.
                 Use 30+ for single-channel (default: 30).
    delay      : Guard interval in frames — prevents direct-sound cancellation (default: 3).
    iterations : Offline EM refinement passes (default: 15).
    """

    def __init__(self, taps: int = 30, delay: int = 3, iterations: int = 15):
        self._taps       = taps
        self._delay      = delay
        self._iterations = iterations
        print(f"  NaraWPE  taps={taps}  delay={delay}  iterations={iterations}")

    @property
    def name(self) -> str:
        return f"nara_wpe_taps{self._taps}_delay{self._delay}_iter{self._iterations}"

    def dereverberate(self, reverb_wav: np.ndarray) -> np.ndarray:
        from nara_wpe.wpe import wpe
        from nara_wpe.utils import stft, istft

        y = reverb_wav[np.newaxis, :]                        # (1, samples)
        Y = stft(y, size=512, shift=128)                     # (frames, freq_bins, 1)
        Y = Y.transpose(2, 0, 1)                             # (1, freq_bins, frames)
        Z = wpe(Y, taps=self._taps, delay=self._delay, iterations=self._iterations)
        z = istft(Z.transpose(1, 2, 0), size=512, shift=128) # (1, samples)
        return z[0].astype(np.float32)                       # (samples,)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dereverb_model(args, device: torch.device) -> DereverbModel:
    """Instantiate the dereverberation model selected by args.dereverb_model."""
    if args.dereverb_model == "unet":
        if not args.dereverb_model_dir:
            raise ValueError("--dereverb_model_dir is required when --dereverb_model=unet")
        return UNetDereverbModel(args.dereverb_model_dir, device)
    elif args.dereverb_model == "nara_wpe":
        return NaraWPEModel(args.wpe_taps, args.wpe_delay, args.wpe_iterations)
    else:
        raise ValueError(f"Unknown dereverb model: {args.dereverb_model!r}")
