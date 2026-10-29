"""
Reference store for evaluation: the test-split RIRs at their native length, with the
per-pair metadata every metric and plot needs, written once per split.

A store lives at data/references/<split_id>/test_references.npz beside a reference_config.json
recording what produced it. The split_id is hashed from the test split's own pairs and the
digest of the mapping CSV, so every model trained on the same split shares one store, and a
rebuilt mapping gets a new one.

Only the test split is stored, since only it is ever scored. Which pairs a run held in each of
its three splits is recorded separately, by save_split_identities, in the run's splits.csv.gz.

Run standalone to build a store ahead of time:
    python3 evaluation/build_test_references.py --run-dir outputs/finished/<run>
    python3 evaluation/build_test_references.py --config ~/Projects/Gen-RIR-Baselines/experiments/mp3d_v2/config.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.soundspaces import create_soundspaces_datasets
from utils.io_utils import _fingerprint, compute_split_id
from utils.misc import get_full_path

STORE_NAME  = 'test_references.npz'
CONFIG_NAME = 'reference_config.json'
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / 'data' / 'references'


# =============================================================================
# Split resolution
# =============================================================================

def build_reference_dataset(run_config: dict):
    """
    The test split a run was given, serving RIRs at their native length.

    Every setting that shapes the split is taken from run_config['args'], so the pairs match
    the model's own test loader. Images are not loaded and sample_max_sec is None, neither of
    which moves a row: the split is decided by the mapping columns alone.

    Args:
        run_config: a run's run_config.json, or an experiment config of the same shape

    Returns:
        A SoundSpacesDataset over the test split.
    """
    a = run_config['args']
    _, _, test_dataset = create_soundspaces_datasets(
        scene_dataset=a['scene_dataset'],
        mapping_csv=os.path.expanduser(a['mapping_csv']),
        geometry_csv=os.path.expanduser(a['geometry_csv']),
        rir_root=a['data_path'],
        image_root=None,
        rir_view_type=None,
        room_overview_type=None,
        max_pairs_per_room=a['max_pairs_per_room'],
        balance_split_by=a['balance_split_by'],
        min_rt60=a['min_rt60'],
        max_scaled_peak=a['max_scaled_peak'],
        max_void_indoor=a['max_void_indoor'],
        max_void_outdoor=a['max_void_outdoor'],
        max_room_void=a['max_room_void'],
        scenes=a['scenes'],
        split=True,
        split_by_room=a['split_by_room'],
        train_ratio=a['train_ratio'],
        eval_ratio=a['eval_ratio'],
        test_ratio=a['test_ratio'],
        random_seed=a['random_seed'],
        sample_max_sec=None,
        sr_target=a['sr_target'],
        use_rt60=a['use_rt60_condition'],
    )
    return test_dataset


def resolve_split_id(run_config: dict, test_dataset) -> tuple:
    """
    The split_id for a test split, and the mapping-CSV fingerprint behind it.

    A run that already recorded a split_id is trusted only after the recomputed value agrees
    with it, so a store can never be attributed to the wrong split.

    Returns:
        (split_id, mapping fingerprint dict)
    """
    # now() makes the fingerprint flag a mapping CSV being rewritten while the store is built
    fingerprint = _fingerprint(os.path.expanduser(run_config['args']['mapping_csv']), datetime.now())
    split_id = compute_split_id(test_dataset.mapping, fingerprint['md5'])

    recorded = run_config.get('split_id')
    if recorded and recorded != split_id:
        raise ValueError(f"split_id mismatch: the run recorded {recorded}, this split hashes to "
                         f"{split_id}. The mapping CSV or a split setting has changed since training.")
    return split_id, fingerprint


# =============================================================================
# Store contents
# =============================================================================

def collect_metadata(test_dataset) -> dict:
    """
    Per-pair columns read straight off the mapping and geometry tables, in mapping row order.

    rt60, c50 and drr are the precomputed columns, left at the float64 the CSV holds them in.
    They were measured with the same functions an evaluation uses, on the same channel of the
    same native-length signal, so they are a cache rather than a second measurement. room_type
    comes from the geometry table's room label.

    Returns:
        A dict of 1-D arrays, each of length len(test_dataset).
    """
    mapping = test_dataset.mapping
    scene = mapping['scene'].astype(str).to_numpy()
    recv  = mapping['receiver_idx'].astype(int).to_numpy()
    src   = mapping['source_idx'].astype(int).to_numpy()

    # The room a pair's receiver sits in, and the label the geometry table gives that room
    room_id = mapping['receiver_region'].astype(int).to_numpy()
    labels  = test_dataset.geometry['label']
    room_type = np.array([str(labels.loc[(s, r)]) for s, r in zip(scene, room_id)])

    return {
        'sample_id':  np.array([f'{s}/{a}_{b}' for s, a, b in zip(scene, recv, src)]),
        'scene':      scene,
        'room_id':    room_id,
        'room_type':  room_type,
        'distance_m': mapping['distance_m'].to_numpy(dtype=np.float64),
        'rt60':       mapping['rt60'].to_numpy(dtype=np.float64),
        'c50':        mapping['c50'].to_numpy(dtype=np.float64),
        'drr':        mapping['drr'].to_numpy(dtype=np.float64),
    }


def _passthrough(batch):
    """Collate that keeps samples as a list, since native-length RIRs cannot be stacked."""
    return batch


def collect_references(test_dataset, workers: int, report_every: int = 2000) -> dict:
    """
    Iterate the test split and gather each pair's native-length RIR and conditioning values.

    Order follows the dataset index, which is mapping row order, so the result aligns with
    collect_metadata. Reads run through DataLoader workers; nothing is stacked.

    The RIRs keep the float32 torchaudio reads them at, and the room-aligned coordinates the
    float32 the dataset builds them as. 'rt60_served' is carried only so check_alignment can
    compare it against the stored column; it is not part of the store.

    Returns:
        A dict with 'references' (object array of 1-D float32), 'sample_id', the 'room_dim',
        'mic_loc' and 'speaker_loc' conditioning, and 'rt60_served'.
    """
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,
                                         num_workers=workers, collate_fn=_passthrough)
    uses_rt60 = test_dataset.use_rt60
    refs, ids, room_dims, mic_locs, speaker_locs, rt60s = [], [], [], [], [], []

    started = time.time()
    for batch in loader:
        for sample in batch:
            rir, room_dim, mic_loc, speaker_loc, _images, _scene, sample_id = sample[:7]
            refs.append(rir.squeeze(0).numpy().astype(np.float32))   # [1, T] -> [T]
            ids.append(sample_id)
            room_dims.append(room_dim)
            mic_locs.append(mic_loc)
            speaker_locs.append(speaker_loc)
            rt60s.append(sample[7] if uses_rt60 else np.nan)
        if len(refs) % report_every < 8:
            print(f'  read {len(refs):,}/{len(test_dataset):,}  ({time.time() - started:.0f}s)')

    print(f'  read {len(refs):,}/{len(test_dataset):,}  ({time.time() - started:.0f}s)')
    out = np.empty(len(refs), dtype=object)
    out[:] = refs
    return {
        'references':  out,
        'sample_id':   np.array(ids),
        'room_dim':    np.asarray(room_dims, dtype=np.float32),
        'mic_loc':     np.asarray(mic_locs, dtype=np.float32),
        'speaker_loc': np.asarray(speaker_locs, dtype=np.float32),
        'rt60_served': np.asarray(rt60s, dtype=np.float64),
    }


def check_alignment(metadata: dict, collected: dict) -> None:
    """
    Confirm the iterated samples are the mapping rows the metadata was read from.

    The ids must agree element-wise, and the rt60 the dataset served must equal the stored
    column, since both claim to describe the same pair.
    """
    ids_meta, ids_read = metadata['sample_id'], collected['sample_id']
    if len(ids_meta) != len(ids_read):
        raise ValueError(f'{len(ids_meta)} mapping rows against {len(ids_read)} samples read')

    differing = np.flatnonzero(ids_meta != ids_read)
    if differing.size:
        i = int(differing[0])
        raise ValueError(f'sample_ids differ at index {i:,}: mapping has {ids_meta[i]}, '
                         f'the dataset served {ids_read[i]} ({differing.size:,} rows differ)')

    served, stored = collected['rt60_served'], metadata['rt60']
    if not np.isnan(served).all():
        gap = np.nanmax(np.abs(served - stored))
        if gap > 1e-5:
            raise ValueError(f'the rt60 served and the rt60 column differ by up to {gap:.2e}')


# =============================================================================
# Store I/O
# =============================================================================

def store_path(split_id: str, root=None) -> Path:
    """Where a split's store lives."""
    return Path(root or DEFAULT_ROOT) / split_id / STORE_NAME


def _write_store(target_dir: Path, arrays: dict, config: dict) -> None:
    """
    Write the store and its config into target_dir.

    Everything is written to a staging directory first and moved into place only once complete,
    so a crash can leave no store rather than a partial one that would load and score.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target_dir.parent, prefix='.staging-'))
    try:
        np.savez(staging / STORE_NAME, **arrays)
        with open(staging / CONFIG_NAME, 'w') as f:
            json.dump(config, f, indent=2)
        shutil.rmtree(target_dir, ignore_errors=True)
        os.replace(staging, target_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def ensure_test_references(run_config: dict, workers: int = 8, root=None, overwrite: bool = False) -> Path:
    """
    The store for a run's test split, built if it is not already there.

    Args:
        run_config: a run's run_config.json, or an experiment config of the same shape
        workers:    DataLoader workers for reading the RIRs
        root:       store root, defaulting to data/references
        overwrite:  rebuild even when a store is present

    Returns:
        Path to the .npz.
    """
    test_dataset = build_reference_dataset(run_config)
    split_id, fingerprint = resolve_split_id(run_config, test_dataset)
    path = store_path(split_id, root)

    if path.exists() and not overwrite:
        print(f'References for split {split_id} already at {path}')
        return path

    print(f'Building references for split {split_id}: {len(test_dataset):,} test pairs')
    metadata  = collect_metadata(test_dataset)
    collected = collect_references(test_dataset, workers)
    check_alignment(metadata, collected)

    # sample_id and rt60_served have served their purpose in check_alignment
    arrays = {**metadata, **{k: v for k, v in collected.items()
                             if k not in ('sample_id', 'rt60_served')}}
    lengths = np.array([len(r) for r in arrays['references']])
    config = {
        'split_id':       split_id,
        'created':        time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_samples':      int(len(lengths)),
        'sample_max_sec': None,
        'sr_target':      run_config['args']['sr_target'],
        'scene_dataset':  run_config['args']['scene_dataset'],
        'mapping_csv':    fingerprint,
        'split_settings': {k: run_config['args'][k] for k in
                           ('max_pairs_per_room', 'balance_split_by', 'min_rt60', 'max_scaled_peak',
                            'max_void_indoor', 'max_void_outdoor', 'max_room_void', 'scenes',
                            'split_by_room', 'train_ratio', 'eval_ratio', 'test_ratio', 'random_seed')},
        'native_length_samples': {'min': int(lengths.min()), 'median': int(np.median(lengths)),
                                  'max': int(lengths.max())},
    }

    _write_store(path.parent, arrays, config)
    print(f'Stored {len(lengths):,} references at {path}')
    return path


def load_test_references(split_id: str, root=None) -> dict:
    """
    Read a store back.

    Returns:
        A dict of the stored arrays. 'references' is an object array of 1-D float32, each at its
        own native length.
    """
    path = store_path(split_id, root)
    if not path.exists():
        raise FileNotFoundError(f'no reference store for split {split_id} at {path}')
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Build the evaluation reference store for a test split')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=str, help='Run directory, or a bare folder name under outputs/finished/')
    source.add_argument('--config', type=str, help="A run_config.json or an experiment config of the same shape")
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers for reading RIRs')
    parser.add_argument('--root', type=str, default=None, help='Store root (default: data/references)')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild even if a store is present')

    args = parser.parse_args()
    if args.run_dir:
        args.run_dir = get_full_path(args.run_dir, 'outputs/finished')
    return args


def main():
    args = parse_args()

    # ---------- Load the config that defines the split ----------
    config_path = Path(args.run_dir) / 'run_config.json' if args.run_dir else Path(os.path.expanduser(args.config))
    with open(config_path) as f:
        run_config = json.load(f)
    print(f'Config: {config_path}')
    print(f'  dataset {run_config["args"]["scene_dataset"]}  sr_target {run_config["args"]["sr_target"]}  '
          f'seed {run_config["args"]["random_seed"]}  recorded split_id {run_config.get("split_id")}')

    # ---------- Build or reuse the store ----------
    path = ensure_test_references(run_config, workers=args.workers, root=args.root, overwrite=args.overwrite)

    # ---------- Report what it holds ----------
    store = load_test_references(path.parent.name, root=args.root)
    lengths = np.array([len(r) for r in store['references']])
    print(f'\n{len(lengths):,} references  ·  {lengths.sum() / 1e6:.1f}M samples  ·  '
          f'{path.stat().st_size / 1e6:.0f} MB on disk')
    print(f'rooms {len(set(zip(store["scene"], store["room_id"])))}  ·  '
          f'room types {len(set(store["room_type"]))}  ·  scenes {len(set(store["scene"]))}')


if __name__ == '__main__':
    main()
