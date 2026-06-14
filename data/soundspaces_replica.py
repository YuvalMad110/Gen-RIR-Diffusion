import json
import os
from math import cos, sin, radians

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.dataset_utils import create_data_splits

# Default dataset paths
_DEFAULT_MAPPING_CSV  = os.path.join(os.path.dirname(__file__), "soundspaces_replica_mapping.csv")
_DEFAULT_GEOMETRY_CSV = os.path.join(os.path.dirname(__file__), "room_geometry.csv")
_DEFAULT_RIR_ROOT     = "/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica"
_DEFAULT_IMAGE_ROOT   = "/dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered"

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


def validate_overview_config(config: dict, scenes: list, image_root: str, room_overview_type: str):
    """
    Validates a room overview config against the resolved scene list and disk contents.

    Checks:
    1. Every scene in `scenes` has an entry in config.
    2. All scenes have the same number of corners (required for uniform N_img across batches).
    3. Every image file referenced by the config exists on disk.

    Raises ValueError or FileNotFoundError on the first failure.
    """
    missing = set(scenes) - set(config.keys())
    if missing:
        raise ValueError(f"room_overview_config missing entries for scenes: {sorted(missing)}")

    counts = {scene: len(config[scene]) for scene in scenes}
    unique_counts = set(counts.values())
    if len(unique_counts) > 1:
        raise ValueError(
            "All scenes must have the same number of corners. Got: "
            + ", ".join(f"{s}={n}" for s, n in sorted(counts.items()))
        )

    ext = 'jpg' if room_overview_type == 'rgb' else 'png'
    for scene in scenes:
        for corner in config[scene]:
            path = os.path.join(image_root, 'room_overviews', scene,
                                f"{corner}_{room_overview_type}.{ext}")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Overview image not found: {path}")


def _load_rgb(path: str, transform) -> torch.Tensor:
    """Load a JPEG/PNG RGB image → ImageNet-normalised [3, H, W] tensor."""
    img = Image.open(path).convert('RGB')
    return transform(img)


def _load_depth(path: str, image_size: tuple) -> torch.Tensor:
    """Load a 16-bit depth PNG → float32 [3, H, W] tensor normalised to [0, 1].

    The single depth channel is repeated 3× so depth images are stackable with
    RGB images. Resize is applied with bilinear interpolation.
    """
    img = Image.open(path)
    depth = np.array(img, dtype=np.float32) / 65535.0   # normalise 16-bit → [0, 1]
    tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    tensor = F.interpolate(tensor, size=image_size, mode='bilinear', align_corners=False)
    return tensor.squeeze(0).repeat(3, 1, 1)  # [3, H, W]


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
        rir_root:             root directory of SoundSpaces binaural RIRs
        image_root:           Replica_rendered/ root, or None to omit all images
        rir_view_type:        per-pair image type: 'rgb', 'depth', or None
        room_overview_type:   sensor type for scene-level overview images: 'rgb', 'depth', or None
        room_overview_config: path to JSON mapping scene → [corner_names] (sensor-agnostic),
                              e.g. {"office_2": ["top_corner1", "mid_corner3"]}
        scenes:               optional list of scene names to restrict the dataset
        split_name:    'train', 'eval', or 'test' — set by factory, not by user
        sample_max_sec: maximum RIR length in seconds (default 2)
        sr_target:     resample RIRs to this rate; None keeps original 44.1 kHz
        image_size:    (H, W) to resize images to (default (392, 518))
    """

    def __init__(self,
                 mapping_csv:          str        = _DEFAULT_MAPPING_CSV,
                 geometry_csv:         str        = _DEFAULT_GEOMETRY_CSV,
                 rir_root:             str        = _DEFAULT_RIR_ROOT,
                 image_root:           str        = _DEFAULT_IMAGE_ROOT,
                 rir_view_type:        str        = 'rgb',
                 room_overview_type:   str        = None,
                 room_overview_config: str        = None,
                 scenes:               list       = None,
                 split_name:    str   = None,
                 sample_max_sec: float = _MAX_SEC,
                 sr_target:     int   = None,
                 image_size:    tuple = _DEFAULT_IMAGE_SIZE):

        self.rir_root       = rir_root
        self.image_root     = image_root
        self.rir_view_type        = rir_view_type
        self.room_overview_type   = room_overview_type
        self.sample_max_sec       = sample_max_sec
        self.sr_orig              = _SR
        self.sr_target            = sr_target if sr_target else _SR
        self.split_name           = split_name
        self._image_size          = image_size
        self._scene_overview_map = {}
        if room_overview_config is not None:
            with open(room_overview_config) as f:
                self._scene_overview_map = json.load(f)

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

        # ---- Images ----
        images = self._load_images(row)

        out = (rir, room_dim, mic_loc, speaker_loc, images)
        return out

    # ------------------------------------------------------------------
    def _load_images(self, row) -> torch.Tensor:
        """Assemble all requested images for one sample into [N_img, 3, H, W], or None."""
        if self.image_root is None:
            return None

        images = []

        if self.rir_view_type is not None:
            subfolder = 'target2source_rgb' if self.rir_view_type == 'rgb' else 'target2source_depth'
            ext       = 'jpg'               if self.rir_view_type == 'rgb' else 'png'
            path = os.path.join(self.image_root, subfolder, row['scene'],
                                f"tgt{int(row['receiver_idx'])}_src{int(row['source_idx'])}.{ext}")
            loader = _load_rgb if self.rir_view_type == 'rgb' else _load_depth
            images.append(loader(path, self.image_transform if self.rir_view_type == 'rgb' else self._image_size))

        if self.room_overview_type is not None and self._scene_overview_map:
            ext     = 'jpg' if self.room_overview_type == 'rgb' else 'png'
            loader  = _load_rgb if self.room_overview_type == 'rgb' else _load_depth
            arg     = self.image_transform if self.room_overview_type == 'rgb' else self._image_size
            corners = self._scene_overview_map[row['scene']]   # KeyError → crash (validated at startup)
            for corner in corners:
                path = os.path.join(self.image_root, 'room_overviews', row['scene'],
                                    f"{corner}_{self.room_overview_type}.{ext}")
                images.append(loader(path, arg))  # FileNotFoundError → crash (validated at startup)

        return torch.stack(images) if images else None

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
                                rir_view_type='rgb',
                                room_overview_type=None,
                                room_overview_config=None,
                                scenes=None,
                                nSamples=None,
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
    # Load overview config and resolve scene list
    scene_overview_map = {}
    if room_overview_config is not None:
        with open(room_overview_config) as f:
            scene_overview_map = json.load(f)
        # If no scenes specified, derive from config keys
        if scenes is None:
            scenes = sorted(scene_overview_map.keys())

    mapping = pd.read_csv(mapping_csv, comment='#')
    mapping = mapping[mapping['valid'] == True]
    if scenes is not None:
        mapping = mapping[mapping['scene'].isin(scenes)]

    # Validate overview config before constructing any datasets
    if room_overview_type is not None and scene_overview_map:
        all_scenes = sorted(mapping['scene'].unique())
        validate_overview_config(scene_overview_map, all_scenes, image_root, room_overview_type)

    indices    = list(range(len(mapping)))
    group_keys = mapping['scene'].tolist()

    shared = dict(mapping_csv=mapping_csv, geometry_csv=geometry_csv,
                  rir_root=rir_root, image_root=image_root,
                  rir_view_type=rir_view_type, room_overview_type=room_overview_type,
                  room_overview_config=room_overview_config, use_rt60=use_rt60, **dataset_kwargs)

    if split:
        train_idx, eval_idx, test_idx = create_data_splits(
            indices, group_keys=group_keys, split_by_group=True,
            train_ratio=train_ratio, eval_ratio=eval_ratio,
            test_ratio=test_ratio, seed=random_seed,
        )

        train_scenes = sorted({group_keys[i] for i in train_idx})
        eval_scenes  = sorted({group_keys[i] for i in eval_idx})
        test_scenes  = sorted({group_keys[i] for i in test_idx})

        train_ds = SoundSpacesReplicaDataset(scenes=train_scenes, split_name='train', **shared)
        eval_ds  = SoundSpacesReplicaDataset(scenes=eval_scenes,  split_name='eval',  **shared)
        test_ds  = SoundSpacesReplicaDataset(scenes=test_scenes,  split_name='test',  **shared)
        return train_ds, eval_ds, test_ds
    else:
        return SoundSpacesReplicaDataset(scenes=scenes, split_name=None, **shared)
