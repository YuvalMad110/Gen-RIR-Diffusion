import hashlib
import json
import os
from collections import defaultdict
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

_DATA_DIR = os.path.dirname(__file__)

_DEFAULTS_CONFIG = os.path.join(os.path.dirname(_DATA_DIR), "config", "dataset_defaults.json")


def _load_dataset_defaults(path: str = _DEFAULTS_CONFIG) -> dict:
    """
    Per-dataset definitions, read from config/dataset_defaults.json.

    Keys beginning with an underscore hold the reasoning behind each setting and are
    dropped here, the same convention the model configs use. The two mapping paths are
    stored relative to this directory so the repository moves as a unit, and are resolved
    against it; the roots outside the repository are stored absolute.
    """
    with open(path) as f:
        config = json.load(f)

    defaults = {}
    for name, entry in config.items():
        if name.startswith('_'):
            continue
        settings = {k: v for k, v in entry.items() if not k.startswith('_')}
        for key in ('mapping_csv', 'geometry_csv'):
            settings[key] = os.path.join(_DATA_DIR, settings[key])
        defaults[name] = settings
    return defaults


DATASET_DEFAULTS = _load_dataset_defaults()

# ImageNet mean/std — required by DA3 ViT-L (inherited from DINOv2 pretraining).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Maximum RIR length used for padding/cropping (seconds)
_MAX_SEC = 2

# Matterport3D renders the same four corner views for every room.
_MP3D_CORNERS = ['corner1', 'corner2', 'corner3', 'corner4']


def _room_keys(mapping, scene_dataset) -> list:
    """Room identifier per row: the scene for Replica, (scene, region) for Matterport3D."""
    if scene_dataset == 'matterport3d':
        return list(zip(mapping['scene'], mapping['receiver_region']))
    return mapping['scene'].tolist()


def _subsample_rooms(mapping, room_keys, max_pairs, seed):
    """
    Keep at most max_pairs rows per room, chosen uniformly at random.

    Each room draws from its own generator, seeded by an MD5 of the room key and the
    run seed, so a room's sample is identical no matter which other scenes or rooms
    are loaded alongside it. Taking the first max_pairs rows instead would be heavily
    biased: the mapping is ordered by receiver, so the head of a large room covers
    only a few per cent of its receivers.
    """
    rows_by_room = defaultdict(list)
    for pos, key in enumerate(room_keys):
        rows_by_room[key].append(pos)

    keep = []
    for key, rows in rows_by_room.items():
        if len(rows) <= max_pairs:
            keep.extend(rows)
        else:
            digest = hashlib.md5(f"{key}|{seed}".encode()).hexdigest()
            rng = np.random.default_rng(int(digest[:16], 16))
            keep.extend(rng.choice(rows, size=max_pairs, replace=False).tolist())
    return mapping.iloc[sorted(keep)].reset_index(drop=True)


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


class SoundSpacesDataset(Dataset):
    """
    Dataset of (RIR, conditioning) pairs from SoundSpaces, for scenes drawn from
    either the Replica or the Matterport3D scene dataset.

    Each sample returns a 7-tuple:
        (rir, room_dim, mic_loc, speaker_loc, images, scene, sample_id)

    or an 8-tuple when use_rt60=True:
        (rir, room_dim, mic_loc, speaker_loc, images, scene, sample_id, rt60)

    where:
        rir         — torch.Tensor [1, T]              mono RIR at sr_target
        room_dim    — np.array     [L_m, W_m, H_m]     long axis, short axis, height
        mic_loc     — np.array     [u, v, z_norm]       receiver in room-aligned coords
        speaker_loc — np.array     [u, v, z_norm]       source in room-aligned coords
        images      — torch.Tensor [N_img, 3, H, W]    stacked images, or None
        scene       — str                               scene name (e.g. 'office_2')
        sample_id   — str                               '<scene>/<receiver_idx>_<source_idx>', unique per pair and stable across runs
        rt60        — float                             RT60 in seconds (8-tuple only)

    N_img = (1 if rir_view_type is not None else 0) + corners_per_room.
    All images are [3, H, W] float32 — RGB is ImageNet-normalised, depth is
    normalised to [0, 1] and repeated 3× to match shape.

    Positions are expressed in room-aligned metres:
        u      = metres along the long room axis
        v      = metres along the short room axis
        z_norm = height above floor (floor = 0)

    Args:
        scene_dataset:        'replica' or 'matterport3d'. Replica keys geometry and overviews by scene;
                              Matterport3D keys them by room, using receiver_region to identify it.
        mapping_df:           mapping table, already filtered and split by the factory
        geometry_csv:         path to room_geometry.csv
        rir_root:             root directory of SoundSpaces binaural RIRs
        image_root:           rendered-images root, or None to omit all images
        rir_view_type:        per-pair image type: 'rgb', 'depth', or None
        room_overview_type:   sensor type for room overview images: 'rgb', 'depth', or None
        room_overview_config: Replica only — path to JSON mapping scene → [corner_names],
                              e.g. {"office_2": ["top_corner1", "mid_corner3"]}
        split_name:           'train', 'eval', or 'test' — set by factory, not by user
        sample_max_sec:       maximum RIR length in seconds (default 2). None returns every RIR at its native length,
                              neither cropped nor zero-padded, which makes a batch ragged — single-sample use only.
        sr_target:            resample RIRs to this rate; None keeps the rate they are stored at
        image_size:           (H, W) to resize images to; required when loading images
        use_rt60:             if True, read the precomputed 'rt60' column from mapping_csv
                              and return it as the last element of the sample tuple.
                              Requires 10_precompute_rt60.py to have been run first.
    """

    def __init__(self,
                 scene_dataset:        str,
                 mapping_df:           "pd.DataFrame",
                 geometry_csv:         str,
                 rir_root:             str,
                 image_root:           str,
                 rir_view_type:        str        = 'rgb',
                 room_overview_type:   str        = None,
                 room_overview_config: str        = None,
                 split_name:           str        = None,
                 sample_max_sec:       float      = _MAX_SEC,
                 sr_target:            int        = None,
                 image_size:           tuple      = None,
                 use_rt60:             bool       = False):

        self.rir_root             = rir_root
        self.image_root           = image_root
        self.scene_dataset        = scene_dataset
        self.rir_view_type        = rir_view_type
        self.room_overview_type   = room_overview_type
        self.sample_max_sec       = sample_max_sec
        self.split_name           = split_name
        self.use_rt60             = use_rt60

        self._loads_images = image_root is not None and (rir_view_type or room_overview_type)
        if self._loads_images and image_size is None:
            raise ValueError("image_size missing from the model config")
        self._image_size     = image_size
        self.image_transform = _make_image_transform(image_size) if self._loads_images else None

        self._scene_overview_map = {}
        if scene_dataset == 'replica' and room_overview_config is not None:
            with open(room_overview_config) as f:
                self._scene_overview_map = json.load(f)

        self.mapping = mapping_df.reset_index(drop=True)

        self.sr_native = torchaudio.info(os.path.join(rir_root, self.mapping.iloc[0]['rir_path'])).sample_rate
        self.sr_target = sr_target or self.sr_native

        index_cols = ['scene', 'region_idx'] if scene_dataset == 'matterport3d' else ['scene']
        self.geometry = pd.read_csv(geometry_csv, comment='#').set_index(index_cols)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        row  = self.mapping.iloc[idx]
        key  = ((row['scene'], row['receiver_region'])
                if self.scene_dataset == 'matterport3d' else row['scene'])
        geom = self.geometry.loc[key]

        # ---- RIR ----
        rir_path = os.path.join(self.rir_root, row['rir_path'])
        waveform, sr_orig = torchaudio.load(rir_path)    # [2, T] stereo
        assert sr_orig == self.sr_native, f"Non-uniform sample rates in the RIR set: {rir_path} is {sr_orig} Hz, others are {self.sr_native} Hz"

        channel = int(row['best_channel'])
        rir = waveform[channel:channel + 1, :]           # [1, T]

        if sr_orig != self.sr_target:
            rir = torchaudio.transforms.Resample(sr_orig, self.sr_target)(rir)

        # None keeps the native length; otherwise every RIR leaves at exactly max_len
        if self.sample_max_sec is not None:
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

        sample_id = f"{row['scene']}/{int(row['receiver_idx'])}_{int(row['source_idx'])}"

        out = (rir, room_dim, mic_loc, speaker_loc, images, row['scene'], sample_id)
        if self.use_rt60:
            out += (float(row['rt60']),)
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

        if self.room_overview_type is not None:
            ext     = 'jpg' if self.room_overview_type == 'rgb' else 'png'
            loader  = _load_rgb if self.room_overview_type == 'rgb' else _load_depth
            arg     = self.image_transform if self.room_overview_type == 'rgb' else self._image_size
            if self.scene_dataset == 'matterport3d':
                room_dir = f"{row['scene']}_r{int(row['receiver_region']):03d}"
                corners  = _MP3D_CORNERS
            else:
                room_dir = row['scene']
                corners  = self._scene_overview_map[row['scene']]
            for corner in corners:
                path = os.path.join(self.image_root, 'room_overviews', room_dir,
                                    f"{corner}_{self.room_overview_type}.{ext}")
                images.append(loader(path, arg))

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


# Which script produces each column a filter needs, so a missing one says where to get it.
_COLUMN_SOURCE = {
    'rt60':               '10_precompute_rt60.py',
    'scaled_peak':        '10_precompute_rt60.py',
    'void_frac':          '05d_void_log.py',
    'outdoor':            '03_compute_mp3d_geometry.py',
    'mean_overview_void': '05d_void_log.py',
}


def _require(mapping, column, threshold_name):
    if column not in mapping.columns:
        raise ValueError(
            f"{threshold_name} is set but column '{column}' is missing from the data. "
            f"Run {_COLUMN_SOURCE[column]} to produce it, or set {threshold_name}=None."
        )


def _room_lookup(mapping, geometry, scene_dataset, column):
    """Per-row values of a room property, taken from the geometry and aligned to mapping."""
    index = ['scene', 'region_idx'] if scene_dataset == 'matterport3d' else ['scene']
    key   = pd.Index(list(zip(mapping['scene'], mapping['receiver_region']))
                     if scene_dataset == 'matterport3d' else list(mapping['scene']))
    return key.map(geometry.set_index(index)[column])


def _apply_quality_filters(mapping, thresholds, scene_dataset, geometry_csv):
    """
    Drop pairs the conditioning cannot describe, or that training cannot survive:

        min_rt60          an RT60 too short to be a room
        max_scaled_peak   an RIR that overflows half precision once scaled, ending the run
        max_void_indoor   an enclosed room's pair view that is mostly empty space
        max_void_outdoor  the same for a room open to the sky, which renders emptier
        max_room_void     a room whose overview images are mostly empty space

    Each threshold runs only when it is set, so a dataset without these measurements is
    untouched.
    """
    if all(v is None for v in thresholds.values()):
        return mapping

    # Rooms too dry to be rooms
    if thresholds['min_rt60'] is not None:
        _require(mapping, 'rt60', 'min_rt60')
        mapping = mapping[mapping['rt60'] >= thresholds['min_rt60']]

    # RIRs whose scaled spectrogram exceeds what half precision holds
    if thresholds['max_scaled_peak'] is not None:
        _require(mapping, 'scaled_peak', 'max_scaled_peak')
        mapping = mapping[mapping['scaled_peak'] <= thresholds['max_scaled_peak']]

    indoor_v  = thresholds['max_void_indoor']
    outdoor_v = thresholds['max_void_outdoor']
    room_void = thresholds['max_room_void']
    if indoor_v is None and outdoor_v is None and room_void is None:
        return mapping

    geometry = pd.read_csv(geometry_csv, comment='#')

    # Pair views that are mostly empty space, at the threshold their room type earns
    if indoor_v is not None or outdoor_v is not None:
        _require(mapping, 'void_frac', 'max_void_indoor/max_void_outdoor')
        _require(geometry, 'outdoor', 'max_void_indoor/max_void_outdoor')
        outdoor = _room_lookup(mapping, geometry, scene_dataset, 'outdoor').fillna(False).values
        limit   = np.where(outdoor, 1.0 if outdoor_v is None else outdoor_v,
                                    1.0 if indoor_v  is None else indoor_v)
        mapping = mapping[mapping['void_frac'].fillna(0.0).values <= limit]

    # Rooms whose overviews give the model nothing to condition on
    if room_void is not None:
        _require(geometry, 'mean_overview_void', 'max_room_void')
        _require(geometry, 'outdoor', 'max_room_void')
        mean    = _room_lookup(mapping, geometry, scene_dataset, 'mean_overview_void').values
        outdoor = _room_lookup(mapping, geometry, scene_dataset, 'outdoor').fillna(False).values
        # Enclosed rooms only: outdoor void is sky, and an unmeasured room is not a failure
        mapping = mapping[outdoor | pd.isna(mean) | (mean <= room_void)]

    return mapping


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_soundspaces_datasets(scene_dataset='matterport3d',
                                mapping_csv=None,
                                geometry_csv=None,
                                rir_root=None,
                                image_root=None,
                                max_pairs_per_room='auto',
                                balance_split_by='auto',
                                min_rt60='auto',
                                max_scaled_peak='auto',
                                max_void_indoor='auto',
                                max_void_outdoor='auto',
                                max_room_void='auto',
                                rir_view_type='rgb',
                                room_overview_type=None,
                                room_overview_config=None,
                                scenes=None,
                                split_by_room=True,
                                split=True,
                                train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15,
                                random_seed=42,
                                use_rt60=False,
                                **dataset_kwargs):
    """
    Factory for SoundSpacesDataset with train/eval/test splits.

    Args:
        scene_dataset:      'replica' or 'matterport3d'. Selects the defaults below, and
                            for Matterport3D groups rows by room rather than by scene.
        max_pairs_per_room: 'auto' takes the dataset's value, None keeps every pair, an
                            integer subsamples rooms holding more than that many RIRs so
                            a few large rooms cannot dominate. Subsampling is seeded per
                            room, so it is identical regardless of what else is loaded.
        balance_split_by:   'auto' takes the dataset's value, 'rirs' holds the sample
                            ratios exactly, 'rooms' holds the room ratios and lets the
                            sample ratios land within a few points.
        min_rt60:           drop pairs whose RT60 falls below this, in seconds. A few milliseconds is an estimation failure rather than a room.
        max_scaled_peak:    drop pairs whose scaled spectrogram peaks above this. Half precision holds nothing past 65,504, and one such sample turns the loss to NaN and ends the run.
        max_void_indoor:    drop pairs whose own view is emptier than this, for rooms that
        max_void_outdoor:   are enclosed and open-to-sky respectively. Outdoor void is mostly sky, which matches the RIR, so it tolerates far more.
        max_room_void:      drop enclosed rooms whose four overviews average emptier than this, which marks a room the scanner never properly covered.
                            Each of these five runs only when it is not None.
        split:              if True, returns (train_ds, eval_ds, test_ds); else one dataset

    Returns:
        If split=True:  (train_dataset, eval_dataset, test_dataset)
        If split=False: single dataset covering all rows
    """
    # Setting defaults (by scene_dataset)
    d = DATASET_DEFAULTS[scene_dataset]
    mapping_csv  = mapping_csv  or d['mapping_csv']
    geometry_csv = geometry_csv or d['geometry_csv']
    rir_root     = rir_root     or d['rir_root']
    image_root   = image_root   or d['image_root']
    if max_pairs_per_room == 'auto':
        max_pairs_per_room = d['max_pairs_per_room']
    if balance_split_by == 'auto':
        balance_split_by = d['balance_split_by']
    thresholds = {k: (d[k] if v == 'auto' else v) for k, v in
                  (('min_rt60', min_rt60), ('max_scaled_peak', max_scaled_peak), ('max_void_indoor', max_void_indoor),
                   ('max_void_outdoor', max_void_outdoor), ('max_room_void', max_room_void))}

    # Replica picks overview corners per scene from a JSON map; Matterport3D uses the
    # same four corners for every room.
    scene_overview_map = {}
    if scene_dataset == 'replica' and room_overview_config is not None:
        with open(room_overview_config) as f:
            scene_overview_map = json.load(f)
        if scenes is None:
            scenes = sorted(scene_overview_map.keys())

    # ---- Sample filtering ----
    mapping = pd.read_csv(mapping_csv, comment='#')
    mapping = mapping[mapping['valid'] == True]
    if scene_dataset == 'matterport3d':
        mapping = mapping[mapping['same_region'] == True]
    if scenes is not None:
        mapping = mapping[mapping['scene'].isin(scenes)]
    if use_rt60:
        mapping = mapping[mapping['rt60'].notna()]
    mapping = _apply_quality_filters(mapping, thresholds, scene_dataset, geometry_csv)
    mapping = mapping.reset_index(drop=True)

    if max_pairs_per_room is not None:
        mapping = _subsample_rooms(mapping, _room_keys(mapping, scene_dataset),
                                   max_pairs_per_room, random_seed)

    if scene_overview_map and room_overview_type is not None:
        validate_overview_config(scene_overview_map, sorted(mapping['scene'].unique()),
                                 image_root, room_overview_type)

    shared = dict(scene_dataset=scene_dataset, geometry_csv=geometry_csv,
                  rir_root=rir_root, image_root=image_root,
                  rir_view_type=rir_view_type, room_overview_type=room_overview_type,
                  room_overview_config=room_overview_config, use_rt60=use_rt60, **dataset_kwargs)

    if not split:
        return SoundSpacesDataset(mapping_df=mapping, split_name=None, **shared)

    train_idx, eval_idx, test_idx = create_data_splits(
        list(range(len(mapping))), group_keys=_room_keys(mapping, scene_dataset),
        split_by_group=split_by_room, balance_by=balance_split_by,
        train_ratio=train_ratio, eval_ratio=eval_ratio,
        test_ratio=test_ratio, seed=random_seed,
    )

    train_ds = SoundSpacesDataset(mapping_df=mapping.iloc[train_idx], split_name='train', **shared)
    eval_ds  = SoundSpacesDataset(mapping_df=mapping.iloc[eval_idx],  split_name='eval',  **shared)
    test_ds  = SoundSpacesDataset(mapping_df=mapping.iloc[test_idx],  split_name='test',  **shared)
    return train_ds, eval_ds, test_ds
