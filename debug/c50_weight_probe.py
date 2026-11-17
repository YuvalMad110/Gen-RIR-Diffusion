"""
Measure what the clarity term would contribute to the gradient, and pick the weight it enters with.

The noise loss is a mean squared error on unit-variance noise; the clarity term is a difference of
decibels. Their values are in unrelated units, so a weight chosen by making the two numbers
comparable means nothing. What can be compared is the gradient each one puts on the model's
parameters, and that is what this measures: one forward pass, each term backpropagated on its own,
and the ratio of the resulting norms. The suggested weights follow, so a term asked to carry a
tenth of the gradient carries a tenth of it.

The measurement is taken from a trained checkpoint. An untrained model predicts noise badly enough
that the clean signal its estimate implies is meaningless, and the weight has to work at the
operating point training actually reaches.

Usage:
  python3 debug/c50_weight_probe.py --run outputs/finished/Oct24_22-00-05_dsief08 --gpu 0
  python3 debug/c50_weight_probe.py --run outputs/finished/Oct24_22-00-05_dsief08 --gpu 0 --batches 40
"""

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from run_train import resolve_c50_loss
from trainer import DiffusionTrainer
from utils.dataset_utils import build_condition_tensor
from utils.inference_data_loading import load_pretrained_model
from utils.signal_proc import waveform_to_spectrogram

# Fractions of the noise loss's gradient the clarity term could be asked to carry
TARGET_SHARES = (0.02, 0.05, 0.10, 0.20, 0.50)


# =============================================================================
# Setup
# =============================================================================

def build_train_dataloader(run_config: dict, batch_size: int, workers: int):
    """The training split and collate the run used, serving waveforms rather than spectrograms.

    Args:
        run_config: The run's run_config.json, whose 'args' carry every setting that shapes the split
        batch_size: Samples per batch
        workers:    Dataloader worker processes

    Returns:
        A DataLoader over the training split, yielding the dict the trainer's forward step reads.
    """
    from data.dataset_collate_fn import scale_and_spectrogram_collate_fn
    from data.soundspaces import create_soundspaces_datasets

    a = run_config['args']
    ie_config = run_config.get('image_encoder_config')

    train_dataset, _, _ = create_soundspaces_datasets(
        scene_dataset=a['scene_dataset'], mapping_csv=a['mapping_csv'],
        geometry_csv=a['geometry_csv'], rir_root=a['data_path'], image_root=a['image_root'],
        max_pairs_per_room=a['max_pairs_per_room'], balance_split_by=a['balance_split_by'],
        min_rt60=a['min_rt60'], max_scaled_peak=a['max_scaled_peak'],
        max_void_indoor=a['max_void_indoor'], max_void_outdoor=a['max_void_outdoor'],
        max_room_void=a['max_room_void'], scenes=a['scenes'], split=True,
        split_by_room=a['split_by_room'], train_ratio=a['train_ratio'],
        eval_ratio=a['eval_ratio'], test_ratio=a['test_ratio'], random_seed=a['random_seed'],
        sample_max_sec=a['sample_max_sec'], sr_target=a['sr_target'],
        rir_view_type=None if a['rir_view_type'] == 'none' else a['rir_view_type'],
        room_overview_type=None if a['room_overview_type'] == 'none' else a['room_overview_type'],
        room_overview_config=a['room_overview_config'],
        image_size=tuple(ie_config['image_size']) if ie_config and 'image_size' in ie_config else None,
        use_rt60=a['use_rt60_condition'],
    )

    collate_fn = scale_and_spectrogram_collate_fn(
        sr=a['sr_target'], db_cutoff=a['db_cutoff'], n_fft=a['n_fft'], hop_length=a['hop_length'],
        scale_rir_flag=a['scale_rir'], use_spectrogram=False,
        apply_zero_tail=a['apply_zero_tail'], use_rt60=a['use_rt60_condition'])

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                        collate_fn=collate_fn, drop_last=True, pin_memory=True)
    print(f'Training split {len(train_dataset):,} samples, {len(loader):,} batches of {batch_size}')
    return loader


def build_term_source(run_config: dict, scheduler, c50_loss: dict):
    """A stand-in carrying the attributes the trainer's two loss methods read.

    The methods are taken from DiffusionTrainer itself so the probe measures the code training
    runs, and the stand-in avoids building the run directories a real trainer creates.

    Args:
        run_config: The run's run_config.json
        scheduler:  The noise scheduler the timesteps are drawn against
        c50_loss:   Resolved clarity settings; its weight is not read here

    Returns:
        An object exposing _noise_loss and _c50_term.
    """
    a = run_config['args']
    source = types.SimpleNamespace(
        noise_scheduler=scheduler, n_fft=a['n_fft'], hop_length=a['hop_length'],
        sr=a['sr_target'], scale_rir=a['scale_rir'], c50_loss=c50_loss,
        loss_weighting=run_config['model_config'].get('loss_weighting'))
    source._noise_loss         = types.MethodType(DiffusionTrainer._noise_loss, source)
    source._denoised_waveforms = types.MethodType(DiffusionTrainer._denoised_waveforms, source)
    source._timestep_weight    = types.MethodType(DiffusionTrainer._timestep_weight, source)
    source._c50_term           = types.MethodType(DiffusionTrainer._c50_term, source)
    return source


# =============================================================================
# Measuring
# =============================================================================

def gradient_norm(parameters) -> float:
    """The 2-norm over every parameter currently carrying a gradient."""
    total = sum(float(p.grad.detach().pow(2).sum()) for p in parameters if p.grad is not None)
    return float(np.sqrt(total))


def measure_batch(batch, model, source, scheduler, device, src_trgt_dist_cond: bool) -> dict:
    """One batch's two gradient norms and the loss values behind them.

    The forward pass repeats the trainer's, in full precision so the two norms are comparable
    without a loss scaler's factor sitting in them.

    Args:
        batch:              A collated training batch
        model:              The trained model, in training mode
        source:             The stand-in exposing the two loss methods
        scheduler:          The noise scheduler
        device:             Device to run on
        src_trgt_dist_cond: Whether the conditioning vector carries the source-target distance

    Returns:
        The two gradient norms and the two loss values, or None when no sample in the batch
        carried a defined clarity on both sides.
    """
    rir       = waveform_to_spectrogram(batch['rir'].to(device),
                                        hop_length=source.hop_length, n_fft=source.n_fft)
    condition = build_condition_tensor(batch, device, src_trgt_dist_cond)
    images    = batch['images'].to(device) if 'images' in batch else None

    noise      = torch.randn_like(rir)
    timesteps  = torch.randint(0, scheduler.num_train_timesteps, (rir.shape[0],),
                               device=device).long()
    noisy_rirs = scheduler.add_noise(rir, noise, timesteps)

    noise_pred = model(noisy_rirs, timesteps, condition, images=images)['sample']
    noise_loss = source._noise_loss(noise_pred, noise, batch)
    c50_term   = source._c50_term(noise_pred, noisy_rirs, timesteps, batch)

    if float(c50_term) == 0.0:
        return None

    parameters = model.get_model_params()

    model.zero_grad(set_to_none=True)
    noise_loss.backward(retain_graph=True)
    noise_grad = gradient_norm(parameters)

    model.zero_grad(set_to_none=True)
    c50_term.backward()
    c50_grad = gradient_norm(parameters)

    model.zero_grad(set_to_none=True)
    return {'noise_grad': noise_grad, 'c50_grad': c50_grad,
            'noise_loss': float(noise_loss), 'c50_error': float(c50_term)}


def collect(dataloader, model, source, scheduler, device, src_trgt_dist_cond: bool,
            batches: int) -> list:
    """Every batch's measurement, reporting as it goes."""
    results, started = [], time.time()

    for batch in dataloader:
        measured = measure_batch(batch, model, source, scheduler, device, src_trgt_dist_cond)
        if measured is None:
            print('  batch skipped: no sample carried a defined clarity on both sides')
            continue
        results.append(measured)
        print(f'  batch {len(results):>3}/{batches}  noise |grad| {measured["noise_grad"]:.4e}  '
              f'clarity |grad| {measured["c50_grad"]:.4e}  clarity error '
              f'{measured["c50_error"]:.3f} dB  [{time.time() - started:.0f}s]')
        if len(results) >= batches:
            break

    return results


# =============================================================================
# Reporting
# =============================================================================

def report(results: list, run_dir: str, batch_size: int) -> str:
    """The measured norms, the weights they imply, and what each column means."""
    noise_grad = np.array([r['noise_grad'] for r in results])
    c50_grad   = np.array([r['c50_grad'] for r in results])
    noise_loss = np.array([r['noise_loss'] for r in results])
    c50_error  = np.array([r['c50_error'] for r in results])
    ratio      = noise_grad / c50_grad

    lines = [
        'C50 LOSS WEIGHT PROBE',
        '',
        f'Run:      {run_dir}',
        f'Measured: {len(results)} batches of {batch_size}',
        '',
        'PER-BATCH QUANTITIES',
        '',
        f'  {"":22}{"median":>12}{"mean":>12}{"std":>12}{"min":>12}{"max":>12}',
    ]
    for name, values in (('noise loss', noise_loss), ('clarity error (dB)', c50_error),
                         ('noise |grad|', noise_grad), ('clarity |grad|', c50_grad),
                         ('ratio', ratio)):
        lines.append(f'  {name:22}{np.median(values):>12.4e}{values.mean():>12.4e}'
                     f'{values.std(ddof=1):>12.4e}{values.min():>12.4e}{values.max():>12.4e}')

    lines += [
        '',
        'SUGGESTED WEIGHT',
        '',
        f'  {"share of noise gradient":28}{"weight":>12}',
    ]
    for share in TARGET_SHARES:
        lines.append(f'  {share:<28.2f}{share * np.median(ratio):>12.4e}')

    lines += [
        '',
        'WHAT THESE ARE',
        '',
        '  noise loss        the mean squared error between the predicted and the added noise, the',
        '                    objective training already runs on.',
        '  clarity error     the weighted mean absolute C50 difference in dB between the RIR the',
        '                    noise estimate implies and the reference, both un-scaled.',
        '  |grad|            the 2-norm over every trainable parameter after backpropagating that',
        '                    term alone. The two share one forward pass, so they are measured at',
        '                    the same point.',
        '  ratio             noise |grad| divided by clarity |grad|. A weight of ratio * s makes',
        '                    the clarity term contribute s times the gradient the noise loss does.',
        '',
        '  The model runs in training mode and full precision: conditioning dropout is active as it',
        '  is in training, and no loss scaler sits in either norm.',
    ]
    return '\n'.join(lines)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--run', required=True, help='run directory holding run_config.json')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=None, help="default: the run's own")
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--min-snr-gamma', type=float, default=5.0)
    parser.add_argument('--eps', type=float, default=1e-10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # ---------- Load the trained model ----------
    model, run_config = load_pretrained_model(args.run, device)
    model.train()
    batch_size = args.batch_size or run_config['args']['batch_size']
    trainable = sum(p.numel() for p in model.get_model_params())
    print(f'Device {device}   trainable parameters {trainable:,}   batch {batch_size}')

    # ---------- Build the training data ----------
    dataloader = build_train_dataloader(run_config, batch_size, args.workers)

    # ---------- Measure ----------
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(num_train_timesteps=run_config['model_config']['n_timesteps'])
    c50_loss  = resolve_c50_loss({'weight': 1.0, 'min_snr_gamma': args.min_snr_gamma,
                                  'eps': args.eps})
    source    = build_term_source(run_config, scheduler, c50_loss)
    print(f'\nMeasuring {args.batches} batches, min_snr_gamma {args.min_snr_gamma} …')
    results = collect(dataloader, model, source, scheduler, device,
                      run_config['model_config']['src_trgt_dist_cond'], args.batches)

    if not results:
        raise SystemExit('No batch carried a defined clarity on both sides')

    # ---------- Render ----------
    text = report(results, args.run, batch_size)
    print('\n' + text)
    path = Path(args.run) / 'c50_weight_probe.txt'
    path.write_text(text + '\n')
    print(f'\nwritten to {path}')


if __name__ == '__main__':
    main()
