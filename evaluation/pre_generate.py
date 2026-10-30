"""
Generate a trained model's RIRs over its test split and save them, so scoring never has to
run the GPU again.

The waveforms land in <run_dir>/generations/<tag>/, with the tag derived from the sampler
settings. Several sampler settings can therefore be generated once each and compared later, and
a metric change costs no GPU time at all.

Only the generated waveforms are saved. The references they are scored against come from the
shared store that build_test_references.py writes, keyed by the same split_id.

--run-dir takes a bare run name, which is looked up under outputs/finished/, or a path to a run
directory anywhere.

    python3 evaluation/pre_generate.py --run-dir Oct24_01-07-43_dsief08
    python3 evaluation/pre_generate.py --run-dir Oct24_01-07-43_dsief08 --num-inference-steps 50 --guidance-scale 2.0
    python3 evaluation/pre_generate.py --run-dir outputs/finished/Oct24_01-07-43_dsief08 --limit 64
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.build_test_references import build_reference_dataset, resolve_split_id
from evaluation.generation_utils import (effective_sampler, generation_dir, generation_tag,
                                         inspect_existing, save_generations)
from utils.dataset_utils import build_condition_tensor
from utils.inference_data_loading import (build_test_dataloader, data_params_from_run_config,
                                          load_pretrained_model)
from utils.misc import get_full_path
from utils.signal_proc import spectrogram_to_waveform, unscale_rir


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a trained model's RIRs over its test split")
    parser.add_argument('--run-dir', type=str, required=True, help='Run directory, or a bare folder name under outputs/finished/')
    parser.add_argument('--num-inference-steps', type=int, default=200, help='Reverse diffusion steps; fewer than the model was trained with forces DDIM')
    parser.add_argument('--guidance-scale', type=float, default=2.0, help='Classifier-free guidance scale')
    parser.add_argument('--use-ddim', action='store_true', help='Use DDIM; fewer steps than trained forces it anyway')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the sampling noise')
    parser.add_argument('--batch-size', type=int, default=72)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None, help='Generate only the first N samples, for a smoke test')
    parser.add_argument('--overwrite', action='store_true', help='Replace a set generated with different settings')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    args.run_dir = get_full_path(args.run_dir, 'outputs/finished')
    return args


# =============================================================================
# Run settings
# =============================================================================

def resolve_unscaling(run_config: dict) -> str:
    """
    How this run's generated RIRs must be un-scaled, as recorded at training time.

    A run from before the field was recorded is read the same way save_run_config derives it, so
    the answer comes from what the run did rather than from which dataset it used.

    Returns:
        'condition_rt60' or 'none'.
    """
    recorded = run_config.get('unscaling')
    if recorded:
        return recorded

    derived = 'condition_rt60' if run_config['args'].get('scale_rir') else 'none'
    print(f"  run_config records no 'unscaling'; derived {derived!r} from scale_rir="
          f"{run_config['args'].get('scale_rir')}")
    return derived


def check_batch_matches_model(batch, model, run_config: dict, settings: dict, device,
                              src_trgt_dist_cond: bool) -> None:
    """
    Confirm one batch carries what the model and the un-scaling need, before any GPU time is spent.

    The condition vector must be exactly as wide as the model was built for, and un-scaling by
    the conditioning rt60 needs that rt60 in the batch.
    """
    expected = run_config['model_config']['input_cond_dim']
    width = build_condition_tensor(batch, device, src_trgt_dist_cond).shape[1]
    if width != expected:
        raise ValueError(f'the batch builds a {width}-wide condition but the model takes {expected}. '
                         f'src_trgt_dist_cond={src_trgt_dist_cond} may not match the trained model.')

    if settings['unscaling'] == 'condition_rt60' and 'rt60' not in batch:
        raise ValueError("un-scaling needs the conditioning rt60, which this batch does not carry. "
                         "The run records unscaling='condition_rt60' but use_rt60_condition is off.")

    if model.image_encoder is not None and 'images' not in batch:
        raise ValueError('the model has an image encoder but the batch carries no images; '
                         'check rir_view_type and room_overview_type in the run config.')

    print(f'  batch ok: condition {width}-wide, '
          f"rt60 {'present' if 'rt60' in batch else 'absent'}, "
          f"images {tuple(batch['images'].shape) if 'images' in batch else 'none'}")


def collect_settings(args, run_config: dict, model, split_id: str) -> dict:
    """
    The settings that decide the waveforms, as SETTINGS_KEYS names them.

    The checkpoint's size and mtime stand in for its identity, so a model retrained in place is
    caught without hashing gigabytes.
    """
    checkpoint = Path(args.run_dir) / 'model_best.pth.tar'
    stat = checkpoint.stat()
    sampler = effective_sampler(args.use_ddim, args.num_inference_steps, model.n_timesteps)

    return {
        'split_id':             split_id,
        'sampler':              sampler,
        'num_inference_steps':  args.num_inference_steps,
        'guidance_scale':       args.guidance_scale,
        'seed':                 args.seed,
        'batch_size':           args.batch_size,
        'limit':                args.limit,
        'unscaling':            resolve_unscaling(run_config),
        'sr_target':            run_config['args']['sr_target'],
        'checkpoint_size_bytes': stat.st_size,
        'checkpoint_modified':  datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds'),
    }


# =============================================================================
# Generation
# =============================================================================

def generate_batch(model, batch, settings: dict, device, src_trgt_dist_cond: bool, run_config: dict):
    """
    Generate one batch of RIRs and return them as waveforms at the model's output length.

    The spectrogram the model produces is converted to a waveform and then un-scaled, so the
    result is on the same footing as a reference RIR.

    Returns:
        [B, T] float32 numpy array.
    """
    a = run_config['args']
    conditions = build_condition_tensor(batch, device, src_trgt_dist_cond)
    images = batch['images'].to(device) if 'images' in batch else None
    shape = (conditions.shape[0], 2, *model.sample_size)   # real and imaginary channels

    specs = model.generate(cond=conditions, shape=shape, images=images,
                           num_inference_steps=settings['num_inference_steps'],
                           guidance_scale=settings['guidance_scale'],
                           use_ddim=settings['sampler'] == 'ddim', verbose=False)

    waveforms = spectrogram_to_waveform(specs, a['hop_length'], a['n_fft']).float()   # [B, T]
    if settings['unscaling'] == 'condition_rt60':
        waveforms = unscale_rir(waveforms, a['sr_target'], batch['rt60'].cpu())

    return waveforms.cpu().numpy().astype(np.float32)


def generate_test_split(model, dataloader, settings: dict, device, src_trgt_dist_cond: bool,
                        run_config: dict, limit=None) -> tuple:
    """
    Generate over the test split in dataloader order.

    The global RNG is seeded once before the first batch. That reproduces a whole pass, but only
    for the same batch size and ordering, which is why batch_size is part of the recorded
    settings. GPU kernels may still differ in the last bits between machines.

    Returns:
        (sample_ids, [N, T] waveforms)
    """
    torch.manual_seed(settings['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(settings['seed'])

    # A limited run stops early, so the bar counts only the batches it will actually reach
    batches = len(dataloader) if not limit else min(len(dataloader), -(-limit // dataloader.batch_size))

    ids, waveforms, started = [], [], time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating', total=batches):
            waveforms.append(generate_batch(model, batch, settings, device, src_trgt_dist_cond, run_config))
            ids.extend(batch['sample_id'])
            if limit and len(ids) >= limit:
                break

    generated = np.concatenate(waveforms, axis=0)
    if limit:
        ids, generated = ids[:limit], generated[:limit]

    elapsed = time.time() - started
    print(f'  generated {len(ids):,} RIRs of {generated.shape[1]:,} samples in {elapsed / 60:.1f} min '
          f'({elapsed / max(len(ids), 1):.2f}s each)')
    return ids, generated


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---------- Load the model and the split it was trained against ----------
    model, run_config = load_pretrained_model(args.run_dir, device)
    test_dataset = build_reference_dataset(run_config)
    split_id, _ = resolve_split_id(run_config, test_dataset)

    # ---------- Resolve what this run would generate, and whether it already exists ----------
    settings = collect_settings(args, run_config, model, split_id)
    tag = generation_tag(settings['sampler'], settings['num_inference_steps'],
                         settings['guidance_scale'], settings['seed'], settings['limit'])
    target_dir = generation_dir(args.run_dir, tag)

    print(f'\nRun {Path(args.run_dir).name}  ·  split {split_id}  ·  tag {tag}')
    print('  ' + '  '.join(f'{k}={settings[k]}' for k in
                           ('sampler', 'num_inference_steps', 'guidance_scale', 'seed', 'batch_size', 'unscaling')))

    state, detail = inspect_existing(target_dir, settings)
    if args.overwrite and state != 'absent':
        print(f'  --overwrite: replacing the {state} set already there')
    elif state == 'match':
        print(f'\nAlready generated at {target_dir} ({detail})')
        return
    elif state in ('differs', 'partial'):
        raise SystemExit(f'\n{target_dir} holds a different set and --overwrite was not given.\n  {detail}')

    # ---------- Generate ----------
    data_params = data_params_from_run_config(run_config)
    data_params['nSamples'] = None               # the full test split; --limit is the way to cut it short
    _, dataloader = build_test_dataloader(data_params, args.batch_size, args.workers, run_config)
    src_trgt_dist_cond = run_config['model_config']['src_trgt_dist_cond']

    check_batch_matches_model(next(iter(dataloader)), model, run_config, settings, device, src_trgt_dist_cond)

    print(f'\nGenerating over {len(test_dataset):,} test pairs'
          + (f', stopping at {args.limit:,}' if args.limit else '') + f' on {device}')
    ids, generated = generate_test_split(model, dataloader, settings, device, src_trgt_dist_cond,
                                         run_config, limit=args.limit)

    # ---------- Save ----------
    config = {**settings, 'created': datetime.now().isoformat(timespec='seconds'),
              'run_dir': str(Path(args.run_dir).resolve()), 'n_samples': len(ids),
              'generated_length': int(generated.shape[1]), 'device': str(device),
              'scene_dataset': run_config['args']['scene_dataset']}
    path = save_generations(target_dir, ids, generated, config)
    print(f'\nStored {len(ids):,} generations at {path}  ({path.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
