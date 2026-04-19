import os
from math import cos, sin, radians

import numpy as np
import pandas as pd
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.dataset_utils import create_data_splits

# Default dataset paths
_DEFAULT_MAPPING_CSV  = os.path.join(os.path.dirname(__file__), "soundspaces_replica_mapping.csv")
_DEFAULT_GEOMETRY_CSV = os.path.join(os.path.dirname(__file__), "room_geometry.csv")
_DEFAULT_RIR_ROOT     = "/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica"
_DEFAULT_IMAGE_ROOT   = "/dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered/target2source_rgb"

# (H, W) that preserves the 4:3 aspect ratio of the rendered 1280×960 images
# while keeping both dimensions divisible by 14 (DA3 ViT-L patch size).
# 392 = 28×14, 518 = 37×14  →  518/392 ≈ 1.32 ≈ 4/3.
_DEFAULT_IMAGE_SIZE = (392, 518)

# ImageNet mean/std — required by DA3 ViT-L (inherited from DINOv2 pretraining).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# SoundSpaces binaural RIR sample rate
_SR = 44100

# Maximum RIR length used for padding/cropping (seconds)
_MAX_SEC = 2


def _make_image_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class SoundSpacesReplicaDataset(Dataset):
    """
    Dataset of (RIR, conditioning) pairs from SoundSpaces/Replica.

    Each sample returns a 5-tuple:
        (rir, room_dim, mic_loc, speaker_loc, image)

    where:
        rir         — torch.Tensor [1, T]          mono RIR at 44.1 kHz
        room_dim    — np.array     [L_m, W_m, H_m] long axis, short axis, height
        mic_loc     — np.array     [u, v, z_norm]  receiver in room-aligned coords
        speaker_loc — np.array     [u, v, z_norm]  source in room-aligned coords
        image       — torch.Tensor [1, 3, H, W]    ImageNet-normalised RGB, or None

    Positions are expressed in room-aligned metres:
        u      = metres along the long room axis
        v      = metres along the short room axis
        z_norm = height above floor (floor = 0)

    Args:
        mapping_csv:   path to soundspaces_replica_mapping.csv
        geometry_csv:  path to room_geometry.csv
        rir_root:      root directory of SoundSpaces binaural RIRs
        image_root:    root of rendered RGB images, or None to omit images
        scenes:        optional list of scene names to restrict the dataset
        split_name:    'train', 'eval', or 'test' — set by factory, not by user
        sample_max_sec: maximum RIR length in seconds (default 2)
        sr_target:     resample RIRs to this rate; None keeps original 44.1 kHz
        image_size:    (H, W) to resize images to (default (392, 518))
    """

    def __init__(self,
                 mapping_csv:   str   = _DEFAULT_MAPPING_CSV,
                 geometry_csv:  str   = _DEFAULT_GEOMETRY_CSV,
                 rir_root:      str   = _DEFAULT_RIR_ROOT,
                 image_root:    str   = _DEFAULT_IMAGE_ROOT,
                 scenes:        list  = None,
                 split_name:    str   = None,
                 sample_max_sec: float = _MAX_SEC,
                 sr_target:     int   = None,
                 image_size:    tuple = _DEFAULT_IMAGE_SIZE):

        self.rir_root       = rir_root
        self.image_root     = image_root
        self.sample_max_sec = sample_max_sec
        self.sr_orig        = _SR
        self.sr_target      = sr_target if sr_target else _SR
        self.split_name     = split_name
        self.image_transform = _make_image_transform(image_size)

        # Load mapping (skip comment lines starting with #)
        self.mapping = pd.read_csv(mapping_csv, comment='#')
        if scenes is not None:
            self.mapping = self.mapping[self.mapping['scene'].isin(scenes)].reset_index(drop=True)

        # Load room geometry indexed by scene name
        self.geometry = pd.read_csv(geometry_csv, comment='#').set_index('scene')

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        row  = self.mapping.iloc[idx]
        geom = self.geometry.loc[row['scene']]

        # ---- RIR ----
        rir_path = os.path.join(self.rir_root, row['rir_path'])
        waveform, _ = torchaudio.load(rir_path)          # [2, T] stereo

        channel = int(row['best_channel'])
        rir = waveform[channel:channel + 1, :]           # [1, T]

        if self.sr_orig != self.sr_target:
            rir = torchaudio.transforms.Resample(self.sr_orig, self.sr_target)(rir)

        max_len = int(self.sr_target * self.sample_max_sec)
        rir = torch.nn.functional.pad(rir, (0, max(0, max_len - rir.shape[-1])))
        rir = rir[:, :max_len]

        # ---- Room dimensions ----
        room_dim = np.array([float(geom['L_m']), float(geom['W_m']), float(geom['H_m'])],
                            dtype=np.float32)

        # ---- Positions in room-aligned coords ----
        mic_loc     = self._to_room_coords(
            float(row['receiver_x']), float(row['receiver_y']), float(row['receiver_z']), geom)
        speaker_loc = self._to_room_coords(
            float(row['source_x']),   float(row['source_y']),   float(row['source_z']),   geom)

        # ---- RGB image ----
        image = None
        if self.image_root is not None:
            img_path = os.path.join(
                self.image_root, row['scene'],
                f"tgt{int(row['receiver_idx'])}_src{int(row['source_idx'])}.jpg"
            )
            if os.path.isfile(img_path):
                img = Image.open(img_path).convert('RGB')
                image = self.image_transform(img).unsqueeze(0)  # [1, 3, H, W]

        return rir, room_dim, mic_loc, speaker_loc, image

    # ------------------------------------------------------------------
    def _to_room_coords(self, x: float, y: float, z: float, geom) -> np.ndarray:
        """Convert mesh XYZ to room-aligned (u, v, z_norm) in metres."""
        dx    = x - float(geom['center_x'])
        dy    = y - float(geom['center_y'])
        theta = radians(float(geom['rotation_deg']))
        u      =  dx * cos(theta) + dy * sin(theta)
        v      = -dx * sin(theta) + dy * cos(theta)
        z_norm = z - float(geom['z_min'])
        return np.array([u, v, z_norm], dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_soundspaces_datasets(mapping_csv=_DEFAULT_MAPPING_CSV,
                                geometry_csv=_DEFAULT_GEOMETRY_CSV,
                                rir_root=_DEFAULT_RIR_ROOT,
                                image_root=_DEFAULT_IMAGE_ROOT,
                                scenes=None,
                                split=True,
                                train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15,
                                random_seed=42,
                                **dataset_kwargs):
    """
    Factory for SoundSpacesReplicaDataset with scene-level train/eval/test splits.

    Args:
        split: if True, returns (train_ds, eval_ds, test_ds); if False, single dataset

    Returns:
        If split=True:  (train_dataset, eval_dataset, test_dataset)
        If split=False: single dataset covering all rows
    """
    mapping = pd.read_csv(mapping_csv, comment='#')
    if scenes is not None:
        mapping = mapping[mapping['scene'].isin(scenes)]

    indices    = list(range(len(mapping)))
    group_keys = mapping['scene'].tolist()

    if split:
        train_idx, eval_idx, test_idx = create_data_splits(
            indices, group_keys=group_keys, split_by_group=True,
            train_ratio=train_ratio, eval_ratio=eval_ratio,
            test_ratio=test_ratio, seed=random_seed,
        )

        train_scenes = sorted({group_keys[i] for i in train_idx})
        eval_scenes  = sorted({group_keys[i] for i in eval_idx})
        test_scenes  = sorted({group_keys[i] for i in test_idx})

        train_ds = SoundSpacesReplicaDataset(
            mapping_csv=mapping_csv, geometry_csv=geometry_csv,
            rir_root=rir_root, image_root=image_root,
            scenes=train_scenes, split_name='train', **dataset_kwargs)
        eval_ds  = SoundSpacesReplicaDataset(
            mapping_csv=mapping_csv, geometry_csv=geometry_csv,
            rir_root=rir_root, image_root=image_root,
            scenes=eval_scenes,  split_name='eval',  **dataset_kwargs)
        test_ds  = SoundSpacesReplicaDataset(
            mapping_csv=mapping_csv, geometry_csv=geometry_csv,
            rir_root=rir_root, image_root=image_root,
            scenes=test_scenes,  split_name='test',  **dataset_kwargs)

        return train_ds, eval_ds, test_ds
    else:
        return SoundSpacesReplicaDataset(
            mapping_csv=mapping_csv, geometry_csv=geometry_csv,
            rir_root=rir_root, image_root=image_root,
            scenes=scenes, split_name=None, **dataset_kwargs)
