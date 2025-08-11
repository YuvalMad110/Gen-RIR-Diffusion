from data.gtu_rir import create_gtu_datasets, GTURIRDataset
import numpy as np
import torch
import librosa
import pywt
# from datasets.soundspaces_rir import SoundspacesRIRDataset (future)

def load_rir_dataset(name, path, split=True, nSamples=None, train_ratio=0.7, eval_ratio=0.15, 
                    test_ratio=0.15, random_seed=42, split_by_room=True,
                    mode='raw', sample_max_sec=2, hop_length=256, n_fft=512, use_spectrogram=False, sr_target=None):
    """
    Load RIR dataset(s).
    
    Args:
        name: Dataset name ('gtu')
        path: Path to dataset file
        split: If True, return (train_dataset, eval_dataset, test_dataset). If False, return single dataset.
        nSamples: Number of samples to use from the dataset, None for all
        train_ratio: Proportion for training (default: 0.7)
        eval_ratio: Proportion for evaluation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        random_seed: Random seed for reproducible splits
        split_by_room: If True, split by room IDs to avoid data leakage
        mode: Processing mode (from original GTURIRDataset)
        sample_max_sec: Maximum length in seconds
        hop_length: Hop length for audio processing
        n_fft: FFT size
        use_spectrogram: Whether to use spectrogram
        sr_target: Target sampling rate
        
    Returns:
        If split=True: tuple (train_dataset, eval_dataset, test_dataset)
        If split=False: single dataset
    """
    if name == 'gtu':
        return create_gtu_datasets(
            tar_path=path,
            split=split,
            nSamples=nSamples,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            split_by_room=split_by_room,
            mode=mode,
            sample_max_sec=sample_max_sec,
            hop_length=hop_length,
            n_fft=n_fft,
            use_spectrogram=use_spectrogram,
            sr_target=sr_target
        )
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