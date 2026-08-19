"""
Fourier Feature Positional Encoder for scalar conditioning vectors.

Following Tancik et al. (2020) "Fourier Features Let Networks Learn High
Frequency Functions in Low Dimensional Domains" and INRAS (Li et al., 2022).
"""

import torch
import torch.nn as nn
from typing import List, Optional


_ENCODE_MODES = {'full'}  # extend later: 'locations' ([3:9]), 'spatial' ([0:9]), etc.


class FourierFeatureEncoder(nn.Module):
    """Fourier feature encoder for a scalar condition vector.

    Each encoded scalar x is lifted to
        [sin(f_1 * x_norm), cos(f_1 * x_norm), ..., sin(f_n * x_norm), cos(f_n * x_norm)]
    before the MLP, enabling the network to learn high-frequency functions of the input.

    Args:
        cond_dim:      Total number of scalars in the condition vector.
        n_freqs:       Number of frequency bands per encoded scalar.
        encode_mode:   Which scalars to encode. Currently only 'full' (all scalars).
        include_raw:   If True, concatenate the normalized raw scalar alongside its
                       Fourier features, giving the MLP both a direct global summary
                       and a multi-frequency representation.
        log_scale:     If True, use log-spaced frequencies 2^linspace(0, max_freq_log2, n_freqs)
                       (NeRF style). If False, use linear spacing.
        max_freq_log2: Highest frequency exponent. Finest scale captured = 1 / 2^max_freq_log2
                       of the normalized input range.
        input_norm:    Per-feature scale factors (length >= cond_dim). Each input x[i] is
                       divided by input_norm[i] before encoding to bring raw physical values
                       (metres, seconds) to ~[0, 1]. None = no normalization.
    """

    def __init__(
        self,
        cond_dim: int,
        n_freqs: int = 6,
        encode_mode: str = 'full',
        include_raw: bool = True,
        log_scale: bool = True,
        max_freq_log2: float = 4.0,
        input_norm: Optional[List[float]] = None,
    ):
        super().__init__()
        if encode_mode not in _ENCODE_MODES:
            raise ValueError(f"encode_mode must be one of {_ENCODE_MODES}, got '{encode_mode}'")

        self.cond_dim = cond_dim
        self.n_freqs = n_freqs
        self.encode_mode = encode_mode
        self.include_raw = include_raw
        self.log_scale = log_scale
        self.max_freq_log2 = max_freq_log2
        self.encode_indices: List[int] = _resolve_indices(encode_mode, cond_dim)
        self.n_encoded = len(self.encode_indices)

        if log_scale:
            freqs = 2.0 ** torch.linspace(0.0, max_freq_log2, n_freqs)
        else:
            freqs = torch.linspace(1.0, 2.0 ** max_freq_log2, n_freqs)
        self.register_buffer('freqs', freqs)

        if input_norm is not None:
            norm = torch.tensor(input_norm[:cond_dim], dtype=torch.float32)
            norm = torch.clamp(norm, min=1e-8)
        else:
            norm = torch.ones(cond_dim, dtype=torch.float32)
        self.register_buffer('input_norm_buf', norm)

    def to_config(self) -> dict:
        """Return the complete configuration dict (all fields, including defaults).

        Stored in the model's run_config so the encoder can be reproduced exactly
        even if default values in this class change in the future.
        """
        return {
            'enable': True,
            'n_freqs': self.n_freqs,
            'encode_mode': self.encode_mode,
            'include_raw': self.include_raw,
            'log_scale': self.log_scale,
            'max_freq_log2': self.max_freq_log2,
            'input_norm': self.input_norm_buf.tolist(),
        }

    @property
    def output_dim(self) -> int:
        fourier_dim = self.n_encoded * 2 * self.n_freqs
        if self.include_raw:
            return self.cond_dim + fourier_dim
        # include_raw=False: non-encoded dims pass through raw, encoded dims replaced
        return (self.cond_dim - self.n_encoded) + fourier_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, cond_dim]
        Returns:
            [B, output_dim]
        """
        x_norm = x / self.input_norm_buf  # [B, cond_dim]

        # Select scalars to encode
        to_encode = x_norm if self.encode_mode == 'full' else x_norm[:, self.encode_indices]

        # Fourier features
        proj = to_encode.unsqueeze(-1) * self.freqs      # [B, n_encoded, n_freqs]
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [B, n_encoded, 2*n_freqs]
        fourier = fourier.flatten(1)                     # [B, n_encoded * 2 * n_freqs]

        if self.include_raw:
            return torch.cat([x_norm, fourier], dim=1)

        if self.n_encoded < self.cond_dim:
            other_idx = sorted(set(range(self.cond_dim)) - set(self.encode_indices))
            return torch.cat([x_norm[:, other_idx], fourier], dim=1)
        return fourier


def _resolve_indices(encode_mode: str, cond_dim: int) -> List[int]:
    if encode_mode == 'full':
        return list(range(cond_dim))
    raise ValueError(f"Unknown encode_mode: '{encode_mode}'")
