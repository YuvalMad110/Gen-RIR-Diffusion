import os
import json
import argparse
from datetime import datetime
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
import sys
# Make sure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from data.soundspaces import DATASET_DEFAULTS, create_soundspaces_datasets
from data.dataset_collate_fn import scale_and_spectrogram_collate_fn
from RIRDiffusionModel import RIRDiffusionModel
from image_encoder import ImageEncoder
from diffusers import DDPMScheduler
from trainer import DiffusionTrainer
from utils.misc import str2bool, filter_threshold, get_git_hash, build_run_header, get_full_path
from utils.epoch_subset_sampler import EpochSubsetSampler
from utils.io_utils import save_run_config, save_split_identities

_DEFAULT_CFG     = 'model_config_VisualCond_MP3D.json'

"""
# Replica — the scene dataset has to be named, since matterport3d is the default:
CUDA_VISIBLE_DEVICES=1 python3 ./Projects/Gen-RIR-Diffusion/run_train.py \
    --batch-size 58 --scene-dataset replica \
    --model-config model_config_VisualCond_HighRes_PosEnc512.json \
    |& tee -a "./Projects/Gen-RIR-Diffusion/outputs/logs/train_$(hostname -s)_$(date +%F_%H-%M-%S).log"

# Smoke test before a long run — one epoch over a meaningful percentage of the dataset,
# with the settings the real run will use:
CUDA_VISIBLE_DEVICES=1 python3 ./Projects/Gen-RIR-Diffusion/run_train.py \
    --batch-size 58 --nSamples 5800 --epochs 1 --post-train-eval false \
    --scene-dataset matterport3d --model-config model_config_VisualCond_MP3D.json
"""

# ------------------------- Utils --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # Model configuration
    parser.add_argument('--model-config', type=str, default=_DEFAULT_CFG,
                        help='Path to model config JSON, relative to config/ or absolute')

    # Dataset selection
    parser.add_argument('--scene-dataset', type=str, default='matterport3d',
                        choices=['replica', 'matterport3d'], help='Scene dataset the SoundSpaces RIRs were simulated in')
    parser.add_argument('--data-path', type=str, default=None,
                        help='SoundSpaces RIR root directory (default: per scene dataset)')
    parser.add_argument('--image-root', type=str, default=None,
                        help="Rendered-images root (default: the dataset's); set --rir-view-type and --room-overview-type to none to disable images")
    parser.add_argument('--rir-view-type', type=str, default='rgb', choices=['rgb', 'depth', 'none'],
                        help="Per-pair image type: 'rgb', 'depth', or 'none' to disable")
    parser.add_argument('--room-overview-type', type=str, default='rgb', choices=['rgb', 'depth', 'none'],
                        help="Sensor type for scene-level overview images; 'none' to disable")
    parser.add_argument('--room-overview-config', type=str, help='Replica only — JSON mapping scene → [corner_names]',
                        default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/config/room_overviews/2views_top_mid_small_scenes.json')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                        help='Restrict to specific scenes (default: all)')
    parser.add_argument('--use-rt60-condition', type=str2bool, default=True,
                        help='Append precomputed RT60 to the SoundSpaces condition vector (requires 10_precompute_rt60.py)')

    # Training configuration
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n-timesteps', type=int, default=1000)
    parser.add_argument('--checkpoint-freq', type=int, default=5)
    parser.add_argument('--eval-freq', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--use-eval', type=str2bool, default=True, help='Use evaluation dataset during training')

    # Audio / spectrogram — all four default to the scene dataset's values
    parser.add_argument('--sr-target', type=int, default=None, help='Target sampling rate')
    parser.add_argument('--sample-max-sec', type=float, default=None, help='Max RIR length in seconds')
    parser.add_argument('--n-fft', type=int, default=None)
    parser.add_argument('--hop-length', type=int, default=None)

    parser.add_argument('--nSamples', type=int, default=None, help='Cap dataset size (None = all)')

    # Quality filters: 'auto' takes the scene dataset's value, 'none' switches one off
    parser.add_argument('--min-rt60', type=filter_threshold, default='auto', help='Drop pairs whose RT60 is below this, in seconds')
    parser.add_argument('--max-scaled-peak', type=filter_threshold, default='auto', help='Drop pairs whose scaled spectrogram peaks above this; 65504 is the half-precision ceiling')
    parser.add_argument('--max-void-indoor', type=filter_threshold, default='auto', help='Drop pairs in enclosed rooms whose own view is emptier than this')
    parser.add_argument('--max-void-outdoor', type=filter_threshold, default='auto', help='Drop pairs in open-to-sky rooms whose own view is emptier than this')
    parser.add_argument('--max-room-void', type=filter_threshold, default='auto', help='Drop enclosed rooms whose four overviews average emptier than this')
    parser.add_argument('--scale-rir', type=str2bool, default=True, help='Flatten each RIR\'s decay at the rate its stored rt60 implies (requires --use-rt60-condition)')
    parser.add_argument('--apply-zero-tail', type=str2bool, default=False, help='Zero RIR values after the db-cutoff point')
    parser.add_argument('--db-cutoff', type=float, default=-40.0, help='dB level the zero-tail cutoff point is read at')

    # Data split
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--eval-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--split-by-room', type=str2bool, default=True, help='Split at scene level to avoid data leakage (default True)')
    parser.add_argument('--random-seed', type=int, default=42)

    # Post-training evaluation
    parser.add_argument('--post-train-eval', type=str2bool, default=True, help='Run full_model_eval.py on the finished run directory after training completes')

    args = parser.parse_args()

    # Resolve the scene-dataset defaults here so run_config.json records the values
    # actually used rather than the Nones that stood in for them.
    d = DATASET_DEFAULTS[args.scene_dataset]
    args.data_path          = args.data_path  or d['rir_root']
    args.image_root         = args.image_root or d['image_root']
    args.mapping_csv        = d['mapping_csv']
    args.geometry_csv       = d['geometry_csv']
    args.max_pairs_per_room = d['max_pairs_per_room']
    args.balance_split_by   = d['balance_split_by']
    for name in ('sr_target', 'sample_max_sec', 'n_fft', 'hop_length'):
        if getattr(args, name) is None:
            setattr(args, name, d[name])
    for name in ('min_rt60', 'max_scaled_peak', 'max_void_indoor', 'max_void_outdoor', 'max_room_void'):
        if getattr(args, name) == 'auto':
            setattr(args, name, d[name])

    if args.scale_rir and not args.use_rt60_condition:
        parser.error('--scale-rir takes its decay rate from the stored rt60, so it requires --use-rt60-condition.')
    return args

def load_model_config(config_path, use_rt60_condition=False):
    """Load and prepare model config. Returns (model_config, ie_config, src_trgt_dist_cond).

    Pops image_encoder_config into ie_config, converts image_size to tuple,
    and extracts src_trgt_dist_cond (consumed here to compute input_cond_dim;
    not a RIRDiffusionModel parameter).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    with open(config_path) as f:
        model_config = json.load(f)
    # Strip comment keys (any key starting with "_" is treated as a comment)
    model_config = {k: v for k, v in model_config.items() if not k.startswith('_')}
    # input_cond_dim: base 9-dim + 1 for RT60 if enabled; +1 if src_trgt_dist_cond
    src_trgt_dist_cond = model_config.pop('src_trgt_dist_cond')
    base_dim = 10 if use_rt60_condition else 9
    model_config['input_cond_dim'] = base_dim + (1 if src_trgt_dist_cond else 0)
    ie_config = model_config.pop('image_encoder_config', None)
    if ie_config and 'image_size' in ie_config:
        ie_config['image_size'] = tuple(ie_config['image_size'])
    loss_weighting = model_config.pop('loss_weighting', None)
    if loss_weighting is not None and loss_weighting.get('method') == 'rt60_binary' and not use_rt60_condition:
        raise ValueError("loss_weighting 'rt60_binary' requires use_rt60_condition=True (batch['rt60'] must be present)")
    return model_config, ie_config, src_trgt_dist_cond, loss_weighting

def num_workers_test(dataset, nWorkers=[0,1,4,8,10, 12, 14, 16,18], batch_size=16):
    """
    Test the performance of different number of workers for DataLoader.
    """
    import time
    print('\n\n*********  Starting num-workers test!!... *********\n\n')
    for nWorker in nWorkers:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nWorker,
            collate_fn=None,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
        start_time = time.time()
        for _ in dataloader:
            pass
        elapsed_time = time.time() - start_time
        print(f"Number of workers: {nWorker}, Time taken: {elapsed_time:.2f} seconds")



def make_train_dataloader(dataset, args, collate_fn):
    """Build the training DataLoader, using EpochSubsetSampler when nSamples is set."""
    if args.nSamples and args.nSamples < len(dataset):
        sampler = EpochSubsetSampler(len(dataset), args.nSamples)
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.workers, collate_fn=collate_fn,
                          drop_last=True, pin_memory=torch.cuda.is_available())
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.workers, collate_fn=collate_fn,
                      drop_last=True, pin_memory=torch.cuda.is_available())


def get_sample_size(dataloader, n_fft, hop_length):
    """Get the sample size that the UNet will operate on (spectrogram dimensions).
    If the dataloader returns spectrograms [B, C, F, T], use those directly.
    If it returns raw waveforms [B, C, T], compute the spectrogram dimensions from n_fft/hop_length.
    """
    for batch in dataloader:
        rir = batch['rir']
        if rir.ndim >= 3:
            return rir.shape[2:]
        # Raw waveforms [B, T] — compute spectrogram size
        n_freq = n_fft // 2 + 1
        n_frames = rir.shape[-1] // hop_length + 1
        return torch.Size([n_freq, n_frames])

def _run_post_train_eval(model_dir: str, batch_size: int) -> None:
    """Launch full_model_eval.py as a subprocess after training completes."""
    import subprocess
    project_root = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(project_root, 'evaluation', 'full_model_eval.py')
    cmd = [
        'python3', eval_script,
        '--model_dir', model_dir,
        '--batch_size', str(batch_size),
    ]
    print(f"\n---------- Post-training evaluation: {' '.join(cmd)} ----------\n")
    subprocess.run(cmd, check=False)


# ------------------------- Main --------------------------
def main():
    started_at = datetime.now()   # inputs written after this are flagged in run_config
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    git_hash = get_git_hash()

    # ---------- debug mode ----------
    debug_mode = False
    if debug_mode:
        print("\nXXXXXXX\n Running in DEBUG mode!!!! \nXXXXXXX\n")
        args.batch_size = 2
        args.epochs = 2
        args.nSamples = 14
        args.split_by_room = False

    # ---------- Model config ----------
    args.model_config = str(get_full_path(args.model_config, "config"))
    model_config, ie_config, src_trgt_dist_cond, loss_weighting = load_model_config(args.model_config, args.use_rt60_condition)

    # ---------- Load datasets ----------
    print("\n----------- Loading datasets... -----------\n")
    rir_view_type      = None if args.rir_view_type      == 'none' else args.rir_view_type
    room_overview_type = None if args.room_overview_type == 'none' else args.room_overview_type
    image_size = ie_config.get('image_size') if ie_config else None
    train_dataset, eval_dataset, test_dataset = create_soundspaces_datasets(
        scene_dataset=args.scene_dataset, rir_root=args.data_path, image_root=args.image_root,
        mapping_csv=args.mapping_csv, geometry_csv=args.geometry_csv,
        max_pairs_per_room=args.max_pairs_per_room, balance_split_by=args.balance_split_by,
        min_rt60=args.min_rt60, max_scaled_peak=args.max_scaled_peak, max_void_indoor=args.max_void_indoor,
        max_void_outdoor=args.max_void_outdoor, max_room_void=args.max_room_void,
        rir_view_type=rir_view_type, image_size=image_size,
        room_overview_type=room_overview_type,
        room_overview_config=args.room_overview_config,
        sample_max_sec=args.sample_max_sec, sr_target=args.sr_target,
        scenes=args.scenes, split_by_room=args.split_by_room,
        train_ratio=args.train_ratio, eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio, random_seed=args.random_seed, use_rt60=args.use_rt60_condition,
    )

    print(f"Dataset splits: {len(train_dataset)} - {len(eval_dataset)} - {len(test_dataset)}")

    # ---------- Create dataloaders ----------
    collate_fn = scale_and_spectrogram_collate_fn(
        sr=args.sr_target, db_cutoff=args.db_cutoff, n_fft=args.n_fft, hop_length=args.hop_length,
        scale_rir_flag=args.scale_rir, use_spectrogram=False, apply_zero_tail=args.apply_zero_tail,
    )
    
    train_dataloader = make_train_dataloader(train_dataset, args, collate_fn)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, drop_last=False, pin_memory=torch.cuda.is_available())

    sample_size = get_sample_size(train_dataloader, args.n_fft, args.hop_length)
    run_header = build_run_header(args, ie_config, git_hash, model_config, sample_size)
    print(f"\nxxxxxxx\nDataloader splits: {len(train_dataloader)} - {len(eval_dataloader)}\nxxxxxxx\n")


    # ---------- Model ----------
    image_encoder = None
    if args.image_root is not None:
        assert ie_config is not None, "image_encoder_config missing from model config JSON"
        cross_attention_dim = model_config['encoder_hidden_dims'][-1]
        image_encoder = ImageEncoder(out_dim=cross_attention_dim, **ie_config)
        print(f"\nImageEncoder: DA3 ViT-{ie_config['model_variant']}, out_dim={cross_attention_dim}, "
              f"~{image_encoder.n_tokens(n_img=1)} tokens per sample\n")

    model = RIRDiffusionModel(
        device=device,
        sample_size=sample_size,
        n_timesteps=args.n_timesteps,
        image_encoder=image_encoder,
        **model_config,
    )
    model.config['src_trgt_dist_cond'] = src_trgt_dist_cond
    model.config['loss_weighting'] = loss_weighting
    print("\n---------- Initialized model ----------\n")
    
    # Scheduler 
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.n_timesteps)
    # Optimizer 
    optimizer = torch.optim.AdamW(model.get_model_params(), lr=args.lr)
    # Accelerator
    accelerator = Accelerator()
    # Trainer
    trainer = DiffusionTrainer(
        device=device,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        accelerator=accelerator,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        run_header=run_header,
        src_trgt_dist_cond=src_trgt_dist_cond,
        sr=args.sr_target,
        loss_weighting=loss_weighting,
    )

    # ---------- Train ----------
    print("\n---------- Starting training... ----------\n")
    model_path = trainer.train(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)

    # ---------- Save run config and model artifacts ----------
    if accelerator.is_main_process:
        configs_dir = os.path.join(model_path, 'configs')

        save_run_config(model_path, args, model.config, ie_config, git_hash, train_dataset, started_at, DATASET_DEFAULTS[args.scene_dataset])
        save_split_identities(model_path, (train_dataset, eval_dataset, test_dataset))

        with open(os.path.join(configs_dir, 'model_config.json'), 'w') as f:
            json.dump(model.config, f, indent=2)

        import shutil
        shutil.copy(args.model_config, os.path.join(configs_dir, 'model_config_original.json'))

        if args.room_overview_config is not None:
            shutil.copy(args.room_overview_config, os.path.join(configs_dir, 'room_overview_config.json'))

        print('\n---------- Training finished successfully!! ----------\n')

        if args.post_train_eval:
            _run_post_train_eval(model_path, args.batch_size)

    # ---------- Distroy Process Group ----------
    # Wait for main process to finish saving files
    accelerator.wait_for_everyone()
    
    # Clean up distributed training (optional but recommended)
    if accelerator.num_processes > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()