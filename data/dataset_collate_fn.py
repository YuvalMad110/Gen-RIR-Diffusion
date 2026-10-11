import torch
from typing import List, Tuple
from utils.signal_proc import waveform_to_spectrogram, scale_rir, zero_tail


def scale_and_spectrogram_collate_fn(sr: float,
                                     db_cutoff: float = -40.0,
                                     n_fft: int = 256,
                                     hop_length: int = 64,
                                     scale_rir_flag: bool = True,
                                     use_spectrogram: bool = True,
                                     apply_zero_tail: bool = False):
    """
    Factory that returns a collate function for use with DataLoader.

    The SoundSpaces dataset guarantees that __getitem__ returns a mono RIR tensor
    of shape [1, T]. The collate assumes this contract.

    The returned function produces a dict batch with keys:
        'rir'         — [B, 2, F, T] spectrogram or [B, 1, T] waveform
        'room_dim'    — [B, 3]
        'mic_loc'     — [B, 3]
        'speaker_loc' — [B, 3]
        'scene'       — list[str]         scene name per sample
        'rt60'        — [B]              when use_rt60=True
        'images'      — [B, N, 3, H, W]  when images are present
                        N = number of images per RIR sample. Currently N=1
                        (one receiver→source RGB per pair); the design supports
                        N>1 for future additions (e.g. overview images).

    Args:
        sr:              sample rate of the RIRs
        db_cutoff:       dB level the zero-tail cutoff point is read at
        n_fft:           FFT size for spectrogram
        hop_length:      hop length for spectrogram
        scale_rir_flag:  whether to flatten each RIR's decay, at the rate its rt60 implies
        use_spectrogram: whether to convert to spectrogram
        apply_zero_tail: whether to zero the signal after the db_cutoff point
    """

    def _process_rirs(rirs, rt60s):
        # Stack mono RIRs [1, T] → [B, 1, T], then squeeze to [B, T] for processing
        batch_rirs = torch.stack(rirs).squeeze(1)  # [B, T]

        # Zeroing runs first, so the cutoff point is read off the natural decay
        if apply_zero_tail:
            batch_rirs = zero_tail(batch_rirs, sr, db_cutoff)

        if scale_rir_flag:
            batch_rirs = scale_rir(batch_rirs, sr, rt60s)

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

    def soundspaces_collate(batch: List[Tuple]) -> dict:
        # Tuple layout: (rir, room_dim, mic_loc, speaker_loc, images, scene, *rt60(*optional))
        if len(batch[0]) == 7:
            rirs, room_dims, mic_locs, speaker_locs, images, scenes, rt60s = zip(*batch)
        elif len(batch[0]) == 6:
            rirs, room_dims, mic_locs, speaker_locs, images, scenes = zip(*batch)
            rt60s = None
        else:
            raise ValueError(
                f"soundspaces_collate: expected 6 or 7 fields, got {len(batch[0])}."
            )
        rt60 = torch.tensor(rt60s, dtype=torch.float32) if rt60s is not None else None

        d = {
            'rir':   _process_rirs(rirs, rt60),
            'scene': list(scenes),
            **_stack_locs(room_dims, mic_locs, speaker_locs),
        }
        if rt60 is not None:
            d['rt60'] = rt60
        # Images may be None when image_root is not set or a file is missing
        if images[0] is not None:
            d['images'] = torch.stack(images)   # [B, N, 3, H, W]
        return d

    return soundspaces_collate
