"""
Measure what the decay-curve term would contribute to the gradient, and pick the weight it enters with.

The noise loss is a mean squared error on unit-variance noise; the decay-curve term is a squared
difference of decibels or of normalised energies. Their values are in unrelated units, so a weight
chosen by making the two numbers comparable means nothing. What can be compared is the gradient each
one puts on the model's parameters, and that is what this measures: one forward pass, each term
backpropagated on its own, and the ratio of the resulting norms. The suggested weights follow, so a
term asked to carry a tenth of the gradient carries a tenth of it.

The two error forms differ by orders of magnitude, so each needs its own measurement and its own
weight. The measurement is taken from a trained checkpoint: an untrained model predicts noise badly
enough that the clean signal its estimate implies is meaningless, and the weight has to work at the
operating point training actually reaches.

Usage:
  python3 debug/edc_weight_probe.py --run outputs/finished/Oct24_22-00-05_dsief08 --gpu 0 --form db
  python3 debug/edc_weight_probe.py --run outputs/finished/Oct24_22-00-05_dsief08 --gpu 0 --form linear
"""

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from run_train import resolve_edc_loss
from trainer import DiffusionTrainer
from utils.dataset_utils import build_condition_tensor
from utils.inference_data_loading import load_pretrained_model
from utils.signal_proc import waveform_to_spectrogram

# Fractions of the noise loss's gradient the decay-curve term could be asked to carry
TARGET_SHARES = (0.02, 0.05, 0.10, 0.20, 0.50)


# =============================================================================
# Setup
# =============================================================================

def stratify_by_rt60(dataset, fraction: float):
    """A subset holding `fraction` of the split, spread evenly over rt60.

    rt60 sets the un-scaling exponent and so the decay shape the term reads, which makes it the one
    variable worth matching when the whole split is too slow to measure. Picking evenly spaced
    positions in the rt60-sorted order reproduces the rt60 distribution with no sampling noise, and
    since rt60 is largely a room property it takes a share of every room's pairs as well.

    Args:
        dataset:  The split to subset
        fraction: Share of the split to keep, in (0, 1]; 1.0 keeps all of it

    Returns:
        The dataset itself at fraction 1.0, otherwise a Subset of it.
    """
    if fraction >= 1.0:
        return dataset
    order = np.argsort(dataset.mapping['rt60'].to_numpy(), kind='stable')
    keep  = max(1, round(len(order) * fraction))
    return Subset(dataset, order[(np.arange(keep) * len(order) // keep)].tolist())


def build_train_dataloader(run_config: dict, batch_size: int, workers: int, fraction: float):
    """The training split and collate the run used, serving waveforms rather than spectrograms.

    Args:
        run_config: The run's run_config.json, whose 'args' carry every setting that shapes the split
        batch_size: Samples per batch
        workers:    Dataloader worker processes
        fraction:   Share of the rt60-sorted split to measure; 1.0 measures all of it

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

    measured = stratify_by_rt60(train_dataset, fraction)
    loader = DataLoader(measured, batch_size=batch_size, shuffle=True, num_workers=workers,
                        collate_fn=collate_fn, drop_last=True, pin_memory=True)
    print(f'Training split {len(train_dataset):,} samples, measuring {len(measured):,} '
          f'({fraction:.0%}, spread over rt60), {len(loader):,} batches of {batch_size}')
    return loader


def build_term_source(run_config: dict, scheduler, edc_loss: dict):
    """A stand-in carrying the attributes the trainer's loss methods read.

    The methods are taken from DiffusionTrainer itself so the probe measures the code training
    runs, and the stand-in avoids building the run directories a real trainer creates.

    Args:
        run_config: The run's run_config.json
        scheduler:  The noise scheduler the timesteps are drawn against
        edc_loss:   Resolved decay-curve settings; its weight is not read here

    Returns:
        An object exposing _noise_loss and _edc_term.
    """
    a = run_config['args']
    source = types.SimpleNamespace(
        noise_scheduler=scheduler, n_fft=a['n_fft'], hop_length=a['hop_length'],
        sr=a['sr_target'], scale_rir=a['scale_rir'], edc_loss=edc_loss,
        loss_weighting=run_config['model_config'].get('loss_weighting'))
    source._noise_loss         = types.MethodType(DiffusionTrainer._noise_loss, source)
    source._denoised_waveforms = types.MethodType(DiffusionTrainer._denoised_waveforms, source)
    source._timestep_weight    = types.MethodType(DiffusionTrainer._timestep_weight, source)
    source._edc_term           = types.MethodType(DiffusionTrainer._edc_term, source)
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
        source:             The stand-in exposing the loss methods
        scheduler:          The noise scheduler
        device:             Device to run on
        src_trgt_dist_cond: Whether the conditioning vector carries the source-target distance

    Returns:
        The two gradient norms and the two loss values, or None when no sample in the batch
        carried a long enough shared window on both sides.
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
    edc_term   = source._edc_term(noise_pred, noisy_rirs, timesteps, batch)

    if float(edc_term) == 0.0:
        return None

    parameters = model.get_model_params()

    model.zero_grad(set_to_none=True)
    noise_loss.backward(retain_graph=True)
    noise_grad = gradient_norm(parameters)

    model.zero_grad(set_to_none=True)
    edc_term.backward()
    edc_grad = gradient_norm(parameters)

    model.zero_grad(set_to_none=True)
    return {'noise_grad': noise_grad, 'edc_grad': edc_grad,
            'noise_loss': float(noise_loss), 'edc_error': float(edc_term)}


def collect(dataloader, model, source, scheduler, device, src_trgt_dist_cond: bool,
            batches: int) -> list:
    """Every batch's measurement, reporting as it goes."""
    results, started = [], time.time()

    for batch in dataloader:
        measured = measure_batch(batch, model, source, scheduler, device, src_trgt_dist_cond)
        if measured is None:
            print('  batch skipped: no sample carried a long enough shared window')
            continue
        results.append(measured)
        elapsed = time.time() - started
        left    = elapsed / len(results) * (batches - len(results))
        print(f'  batch {len(results):>4}/{batches}  noise |grad| {measured["noise_grad"]:.4e}  '
              f'decay |grad| {measured["edc_grad"]:.4e}  decay error '
              f'{measured["edc_error"]:.4e}  [{elapsed / 60:.1f}m elapsed, {left / 60:.1f}m left]')
        if len(results) >= batches:
            break

    return results


# =============================================================================
# Reporting
# =============================================================================

def report(results: list, run_dir: str, batch_size: int, edc_loss: dict) -> str:
    """The measured norms, the weights they imply, and what each column means."""
    noise_grad = np.array([r['noise_grad'] for r in results])
    edc_grad   = np.array([r['edc_grad'] for r in results])
    noise_loss = np.array([r['noise_loss'] for r in results])
    edc_error  = np.array([r['edc_error'] for r in results])
    ratio      = noise_grad / edc_grad

    settings = ', '.join(f'{k} {v}' for k, v in sorted(edc_loss.items()) if k != 'weight')
    unit     = 'dB^2' if edc_loss['form'] == 'db' else 'dimensionless'

    lines = [
        'EDC LOSS WEIGHT PROBE',
        '',
        f'Run:      {run_dir}',
        f'Measured: {len(results)} batches of {batch_size}',
        f'Settings: {settings}',
        '',
        'PER-BATCH QUANTITIES',
        '',
        f'  {"":26}{"median":>12}{"mean":>12}{"std":>12}{"min":>12}{"max":>12}',
    ]
    for name, values in (('noise loss', noise_loss), (f'decay error ({unit})', edc_error),
                         ('noise |grad|', noise_grad), ('decay |grad|', edc_grad),
                         ('ratio', ratio)):
        lines.append(f'  {name:26}{np.median(values):>12.4e}{values.mean():>12.4e}'
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
        '  decay error       the weighted mean squared difference between the energy decay curve of',
        '                    the RIR the noise estimate implies and that of the reference, both',
        '                    un-scaled, peak-anchored and normalised over the window they share.',
        '  |grad|            the 2-norm over every trainable parameter after backpropagating that',
        '                    term alone. The two share one forward pass, so they are measured at',
        '                    the same point.',
        '  ratio             noise |grad| divided by decay |grad|. A weight of ratio * s makes the',
        '                    decay term contribute s times the gradient the noise loss does.',
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
    parser.add_argument('--form', choices=('db', 'linear'), default='db',
                        help='which error form to measure the weight for')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batches', type=int, default=10**9,
                        help='stop after this many batches; the default measures every one')
    parser.add_argument('--batch-size', type=int, default=None, help="default: the run's own")
    parser.add_argument('--fraction', type=float, default=0.2,
                        help='share of the training split to measure, spread over rt60; 1.0 is all of it')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--floor-db', type=float, default=30.0, help="'db' form only")
    parser.add_argument('--min-snr-gamma', type=float, default=5.0)
    parser.add_argument('--min-post-peak-ms', type=float, default=250.0)
    parser.add_argument('--energy-floor', type=float, default=1e-20)
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
    dataloader = build_train_dataloader(run_config, batch_size, args.workers, args.fraction)
    batches = min(args.batches, len(dataloader))

    # ---------- Measure ----------
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(num_train_timesteps=run_config['model_config']['n_timesteps'])
    settings  = {'form': args.form, 'weight': 1.0, 'min_snr_gamma': args.min_snr_gamma,
                 'min_post_peak_ms': args.min_post_peak_ms, 'energy_floor': args.energy_floor}
    if args.form == 'db':
        settings['floor_db'] = args.floor_db
    edc_loss = resolve_edc_loss(settings)
    source   = build_term_source(run_config, scheduler, edc_loss)
    print(f'\nMeasuring {batches} batches, form {args.form} …')
    results = collect(dataloader, model, source, scheduler, device,
                      run_config['model_config']['src_trgt_dist_cond'], batches)

    if not results:
        raise SystemExit('No batch carried a long enough shared window on both sides')

    # ---------- Render ----------
    text = report(results, args.run, batch_size, edc_loss)
    print('\n' + text)
    path = Path(args.run) / f'edc_weight_probe_{args.form}.txt'
    path.write_text(text + '\n')
    print(f'\nwritten to {path}')


if __name__ == '__main__':
    main()
