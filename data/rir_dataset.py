from data.gtu_rir import GTURIRDataset
import numpy as np
import torch
import librosa
import pywt
# from datasets.soundspaces_rir import SoundspacesRIRDataset (future)

def load_rir_dataset(name, path, mode='raw', max_length=88200, use_spectrogram=False, hop_length=256, n_fft=512):
    if name == 'gtu':
        return GTURIRDataset(path, mode=mode, max_length=max_length, use_spectrogram=use_spectrogram, hop_length=hop_length, n_fft=n_fft)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

def collate_fn_raw(batch):
    return torch.stack(batch)

def collate_fn_stft(batch):
    batch_np = torch.stack(batch)
    return torch.stack([
        torch.tensor(np.abs(librosa.stft(x.numpy(), n_fft=512, hop_length=256)), dtype=torch.float32)
        for x in batch_np
    ])

def collate_fn_wavelet(batch):
    batch_np = torch.stack(batch)
    return torch.stack([
        torch.tensor(np.concatenate(pywt.wavedec(x.numpy(), 'db4', level=4)), dtype=torch.float32)
        for x in batch_np
    ])

def get_collate_fn(mode):
    if mode == 'raw':
        return collate_fn_raw
    elif mode == 'stft':
        return collate_fn_stft
    elif mode == 'wavelet':
        return collate_fn_wavelet
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def make_collate_fn(mode):
    def collate_fn(batch):
        batch_np = torch.stack(batch)  # [B, T]

        if mode == 'raw':
            return batch_np.unsqueeze(1)  # [B, 1, T]

        elif mode == 'stft':
            return torch.stack([
                torch.tensor(np.abs(librosa.stft(x.numpy(), n_fft=512, hop_length=256)), dtype=torch.float32)
                for x in batch_np
            ])

        elif mode == 'wavelet':
            return torch.stack([
                torch.tensor(np.concatenate(pywt.wavedec(x.numpy(), 'db4', level=4)), dtype=torch.float32)
                for x in batch_np
            ])

        else:
            raise ValueError(f"Unsupported mode: {mode}")
    return collate_fn
