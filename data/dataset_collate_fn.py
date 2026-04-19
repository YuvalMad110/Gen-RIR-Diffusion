import torch
from typing import List, Tuple
from utils.signal_proc import waveform_to_spectrogram, scale_rir


def scale_and_spectrogram_collate_fn(sr: float,
                                     db_cutoff: float = -40.0,
                                     n_fft: int = 256,
                                     hop_length: int = 64,
                                     scale_rir_flag: bool = True,
                                     use_spectrogram: bool = True,
                                     apply_zero_tail: bool = False,
                                     dataset_type: str = 'gtu'):
    """
    Factory that returns a collate function for use with DataLoader.

    Both GTU and SoundSpaces datasets guarantee that __getitem__ returns a
    mono RIR tensor of shape [1, T]. The collate assumes this contract.

    The returned function produces a dict batch with keys:
        'rir'         — [B, 2, F, T] spectrogram or [B, 1, T] waveform
        'room_dim'    — [B, 3]
        'mic_loc'     — [B, 3]
        'speaker_loc' — [B, 3]
        'rt60'        — [B]              (GTU only)
        'images'      — [B, N, 3, H, W]  (SoundSpaces only, when images are present)
                        N = number of images per RIR sample. Currently N=1
                        (one receiver→source RGB per pair); the design supports
                        N>1 for future additions (e.g. overview images).

    Args:
        sr:              sample rate of the RIRs
        db_cutoff:       dB threshold for EDC-based RIR cropping
        n_fft:           FFT size for spectrogram
        hop_length:      hop length for spectrogram
        scale_rir_flag:  whether to apply RIR scaling
        use_spectrogram: whether to convert to spectrogram
        apply_zero_tail: whether to zero the signal after the db_cutoff point
        dataset_type:    'gtu' or 'soundspaces'
    """

    def _process_rirs(rirs):
        # Stack mono RIRs [1, T] → [B, 1, T], then squeeze to [B, T] for processing
        batch_rirs = torch.stack(rirs).squeeze(1)  # [B, T]

        # Optionally scale the RIRs
        if scale_rir_flag:
            batch_rirs = scale_rir(batch_rirs, sr, db_cutoff, apply_zero_tail)

        # Optionally convert to spectrograms
        if use_spectrogram:
            batch_rirs = waveform_to_spectrogram(
                waveform=batch_rirs, hop_length=hop_length, n_fft=n_fft)

        return batch_rirs

    def _stack_locs(room_dims, mic_locs, speaker_locs):
        return {
            'room_dim':    torch.stack([torch.tensor(r, dtype=torch.float32) for r in room_dims]),
            'mic_loc':     torch.stack([torch.tensor(m, dtype=torch.float32) for m in mic_locs]),
            'speaker_loc': torch.stack([torch.tensor(s, dtype=torch.float32) for s in speaker_locs]),
        }

    def gtu_collate(batch: List[Tuple]) -> dict:
        rirs, room_dims, mic_locs, speaker_locs, rt60s = zip(*batch)
        return {
            'rir':  _process_rirs(rirs),
            **_stack_locs(room_dims, mic_locs, speaker_locs),
            'rt60': torch.tensor(rt60s, dtype=torch.float32),
        }

    def soundspaces_collate(batch: List[Tuple]) -> dict:
        rirs, room_dims, mic_locs, speaker_locs, images = zip(*batch)
        d = {
            'rir': _process_rirs(rirs),
            **_stack_locs(room_dims, mic_locs, speaker_locs),
        }
        # Images may be None when image_root is not set or a file is missing
        if images[0] is not None:
            d['images'] = torch.stack(images)   # [B, N, 3, H, W]
        return d

    if dataset_type == 'gtu':
        return gtu_collate
    elif dataset_type == 'soundspaces':
        return soundspaces_collate
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Expected 'gtu' or 'soundspaces'.")
