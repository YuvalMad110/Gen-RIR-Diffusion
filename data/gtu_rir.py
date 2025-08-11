# datasets/gtu_rir.py (Clean factory function implementation)
import tarfile
import pickle
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from sklearn.model_selection import train_test_split
import hashlib

class GTURIRDataset(Dataset):
    def __init__(self, data, mode='raw', sample_max_sec=2, hop_length=256, n_fft=512, 
                 use_spectrogram=False, sr_orig=44100, sr_target=None, split_name=None):
        """
        Initialize GTU RIR Dataset.
        
        Args:
            data: List of RIR samples (already loaded and filtered)
            split_name: Name of the split ('train', 'eval', 'test', or None for full dataset)
        """
        self.data = data
        self.mode = mode
        self.sample_max_sec = sample_max_sec
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.use_spectrogram = use_spectrogram
        self.sr_orig = sr_orig
        self.sr_target = sr_target if sr_target else sr_orig
        self.split_name = split_name

        self.rir_data_field_numbers = {"timestamp": 0, "speakerMotorIterationNo": 1, "microphoneMotorIterationNo": 2,
                                       "speakerMotorIterationDirection": 3, "currentActiveSpeakerNo": 4,
                                       "currentActiveSpeakerChannelNo": 5,
                                       "physicalSpeakerNo": 6, "microphoneStandInitialCoordinateX": 7,
                                       "microphoneStandInitialCoordinateY": 8, "microphoneStandInitialCoordinateZ": 9,
                                       "speakerStandInitialCoordinateX": 10,
                                       "speakerStandInitialCoordinateY": 11, "speakerStandInitialCoordinateZ": 12,
                                       "microphoneMotorPosition": 13, "speakerMotorPosition": 14,
                                       "temperatureAtMicrohponeStand": 15,
                                       "humidityAtMicrohponeStand": 16, "temperatureAtMSpeakerStand": 17,
                                       "humidityAtSpeakerStand": 18, "tempHumTimestamp": 19,
                                       "speakerRelativeCoordinateX": 20, "speakerRelativeCoordinateY": 21,
                                       "speakerRelativeCoordinateZ": 22, "microphoneStandAngle": 23,
                                       "speakerStandAngle": 24, "speakerAngleTheta": 25, "speakerAnglePhi": 26,
                                       "mic_RelativeCoordinateX": 27, "mic_RelativeCoordinateY": 28,
                                       "mic_RelativeCoordinateZ": 29, "mic_DirectionX": 30, "mic_DirectionY": 31,
                                       "mic_DirectionZ": 32, "mic_Theta": 33, "mic_Phi": 34, "essFilePath": 35,
                                       "roomId": 36, "configId": 37, "micNo": 38,
                                       "roomWidth": 39, "roomHeight": 40, "roomDepth": 41,
                                       "rt60": 42,
                                       "rirData": 43
                                       }

    def get_split_info(self):
        """Return information about the current dataset"""
        return {
            'split': self.split_name or 'full',
            'n_samples': len(self.data)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # ---- PreProcess RIR ----
        rir_np = np.array(sample[43], dtype=np.float32)
        rir = torch.tensor(rir_np).unsqueeze(0)  # [1, T]
        
        # Resample if needed
        if self.sr_orig != self.sr_target:
            resampler = torchaudio.transforms.Resample(
                orig_freq=self.sr_orig,
                new_freq=self.sr_target
            )
            rir = resampler(rir)
            
        # Crop/pad to desired length
        if self.sample_max_sec:
            max_len_samples = int(self.sr_target * self.sample_max_sec)
            rir = torch.nn.functional.pad(rir, (0, max(0, max_len_samples - rir.shape[-1])))
            rir = rir[:, :max_len_samples]

        # ---- Prepare Metadata ----
        # RT60
        rt60 = float(sample[int(self.rir_data_field_numbers['rt60'])])
        
        # Room dimensions [m]
        room_dim = np.array([float(sample[int(self.rir_data_field_numbers['roomDepth'])]),
                             float(sample[int(self.rir_data_field_numbers['roomWidth'])]),
                            float(sample[int(self.rir_data_field_numbers['roomHeight'])])]) / 100 # [depth, width, height] [m]
        
        # Microphone location [m]
        microphone_coordinates_x = float(
            sample[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateX'])]) + float(
            sample[int(self.rir_data_field_numbers['mic_RelativeCoordinateX'])])
        microphone_coordinates_y = float(
            sample[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateY'])])  + float(
            sample[int(self.rir_data_field_numbers['mic_RelativeCoordinateY'])])
        microphone_coordinates_z = float(sample[int(self.rir_data_field_numbers['mic_RelativeCoordinateZ'])])
        mic_loc = np.array([microphone_coordinates_x, microphone_coordinates_y, microphone_coordinates_z]) / 100  # [m]
        
        # Speaker location [m]
        speaker_coordinates_x = float(
            sample[int(self.rir_data_field_numbers['speakerStandInitialCoordinateX'])])  + float(
            sample[int(self.rir_data_field_numbers['speakerRelativeCoordinateX'])])
        speaker_coordinates_y = float(
            sample[int(self.rir_data_field_numbers['speakerStandInitialCoordinateY'])]) + float(
            sample[int(self.rir_data_field_numbers['speakerRelativeCoordinateY'])])
        speaker_coordinates_z = float(sample[int(self.rir_data_field_numbers['speakerRelativeCoordinateZ'])])
        speaker_loc = np.array([speaker_coordinates_x, speaker_coordinates_y, speaker_coordinates_z]) / 100 # [m]

        return rir, room_dim, mic_loc, speaker_loc, rt60


def _load_data_from_tar(tar_path, inside_file='RIR.pickle.dat'):
    """Load data from pickle file"""
    with open(tar_path, 'rb') as f:
        return pickle.load(f)


def _create_data_splits(all_data, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15, 
                       random_seed=42, split_by_room=True):
    """Create train/eval/test data splits"""
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    
    # Define field numbers for room ID access
    rir_data_field_numbers = {"roomId": 36}
    
    if split_by_room:
        # Split by room IDs to prevent data leakage
        room_ids = [sample[rir_data_field_numbers['roomId']] for sample in all_data]
        unique_room_ids = list(set(room_ids))
        
        # Create a deterministic but pseudo-random order based on room IDs
        unique_room_ids.sort()  # Ensure consistent ordering
        room_hash_seeds = [int(hashlib.md5(f"{room_id}_{random_seed}".encode()).hexdigest()[:8], 16) 
                          for room_id in unique_room_ids]
        room_order = np.argsort(room_hash_seeds)
        unique_room_ids = [unique_room_ids[i] for i in room_order]
        
        n_rooms = len(unique_room_ids)
        n_train_rooms = int(n_rooms * train_ratio)
        n_eval_rooms = int(n_rooms * eval_ratio)
        
        train_room_ids = set(unique_room_ids[:n_train_rooms])
        eval_room_ids = set(unique_room_ids[n_train_rooms:n_train_rooms + n_eval_rooms])
        test_room_ids = set(unique_room_ids[n_train_rooms + n_eval_rooms:])
        
        # Filter data based on room IDs
        train_data = [sample for sample in all_data 
                     if sample[rir_data_field_numbers['roomId']] in train_room_ids]
        eval_data = [sample for sample in all_data 
                    if sample[rir_data_field_numbers['roomId']] in eval_room_ids]
        test_data = [sample for sample in all_data 
                    if sample[rir_data_field_numbers['roomId']] in test_room_ids]
        
    else:
        # Random split (may have data leakage if same rooms appear in different splits)
        indices = np.arange(len(all_data))
        
        # First split: train vs (eval + test)
        train_indices, temp_indices = train_test_split(
            indices, test_size=(eval_ratio + test_ratio), 
            random_state=random_seed, shuffle=True
        )
        
        # Second split: eval vs test
        eval_indices, test_indices = train_test_split(
            temp_indices, test_size=test_ratio/(eval_ratio + test_ratio),
            random_state=random_seed, shuffle=True
        )
        
        train_data = [all_data[i] for i in train_indices]
        eval_data = [all_data[i] for i in eval_indices]
        test_data = [all_data[i] for i in test_indices]
    
    return train_data, eval_data, test_data


def create_gtu_datasets(tar_path, split=True, nSamples=None, train_ratio=0.7, eval_ratio=0.15, 
                       test_ratio=0.15, random_seed=42, split_by_room=True, inside_file='RIR.pickle.dat',
                       **dataset_kwargs):
    """
    Factory function to create GTU RIR datasets.
    
    Args:
        tar_path: Path to the pickle file
        split: If True, return (train_dataset, eval_dataset, test_dataset). If False, return single dataset.
        nSamples: Number of samples to use from the dataset, None for all
        train_ratio: Proportion for training (default: 0.7)
        eval_ratio: Proportion for evaluation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        random_seed: Random seed for reproducible splits
        split_by_room: If True, split by room IDs to avoid data leakage
        inside_file: Name of the file inside the tar archive
        **dataset_kwargs: Additional arguments passed to GTURIRDataset constructor
        
    Returns:
        If split=True: (train_dataset, eval_dataset, test_dataset)
        If split=False: single_dataset
    """    
    # Load all data once
    all_data = _load_data_from_tar(tar_path, inside_file)
    
    # Apply nSamples limit if specified
    if nSamples and nSamples < len(all_data):
        all_data = all_data[:nSamples]
    
    if split:
        # Create splits and return 3 datasets
        train_data, eval_data, test_data = _create_data_splits(
            all_data, train_ratio, eval_ratio, test_ratio, random_seed, split_by_room
        )
        
        train_dataset = GTURIRDataset(train_data, split_name='train', **dataset_kwargs)
        eval_dataset = GTURIRDataset(eval_data, split_name='eval', **dataset_kwargs)
        test_dataset = GTURIRDataset(test_data, split_name='test', **dataset_kwargs)
        
        return train_dataset, eval_dataset, test_dataset
    
    else:
        # Return single dataset with all data
        return GTURIRDataset(all_data, split_name=None, **dataset_kwargs)
    