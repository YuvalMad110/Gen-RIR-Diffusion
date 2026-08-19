import os
import json
import argparse
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
import sys
import platform
# Make sure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from data.rir_dataset import load_rir_dataset
from data.dataset_collate_fn import scale_and_spectrogram_collate_fn
from RIRDiffusionModel import RIRDiffusionModel
from image_encoder import ImageEncoder
from diffusers import DDPMScheduler
from trainer import DiffusionTrainer
from utils.misc import str2bool, get_git_hash, build_run_header, get_full_path
from utils.epoch_subset_sampler import EpochSubsetSampler
from utils.io_utils import save_run_config

_DEFAULT_CFG     = 'model_config_VisualCond_HighRes_PosEnc512.json'
_DEFAULT_SS_ROOT = '/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica/'

"""
# GTU (original):
CUDA_VISIBLE_DEVICES=3 python3 ./Projects/Gen-RIR-Diffusion/run_train.py --batch-size 16 --epochs 100 --nSamples 128 \
|& tee -a "./Projects/Gen-RIR-Diffusion/outputs/logs/train_$(hostname -s)_$(date +%F_%H-%M-%S).log"

# SoundSpaces — scalar conditioning only (no images):
CUDA_VISIBLE_DEVICES=1 python3 ./Projects/Gen-RIR-Diffusion/run_train.py \
    --batch-size 58 \
    --model-config model_config_VisualCond_HighRes_NoDist.json \
    |& tee -a "./Projects/Gen-RIR-Diffusion/outputs/logs/train_$(hostname -s)_$(date +%F_%H-%M-%S).log"

"""

# ------------------------- Utils --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # Model configuration
    parser.add_argument('--model-config', type=str, default=_DEFAULT_CFG,
                        help='Path to model config JSON. Use model_config_VisualCond.json for image conditioning.')

    # Dataset selection
    parser.add_argument('--dataset-name', type=str, default='soundspaces', choices=['gtu', 'soundspaces'],
                        help='Dataset to train on. Defaults: gtu→GTU pickle, soundspaces→server RIR root')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Dataset path. Defaults to GTU pickle or SoundSpaces RIR root based on --dataset-name')
    parser.add_argument('--image-root', type=str, default='/dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered/',
                        help='Replica_rendered/ root; omit to disable image conditioning')
    parser.add_argument('--rir-view-type', type=str, default='rgb', choices=['rgb', 'depth', 'none'],
                        help="Per-pair image type: 'rgb', 'depth', or 'none' to disable")
    parser.add_argument('--room-overview-type', type=str, default='rgb', choices=['rgb', 'depth', 'none'],
                        help="Sensor type for scene-level overview images; 'none' to disable")
    parser.add_argument('--room-overview-config', type=str, help='Path to JSON mapping scene → [corner_names]',
                        default='/home/yuvalmad/Projects/Gen-RIR-Diffusion/config/room_overviews/2views_top_mid_small_scenes.json')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                        help='Restrict SoundSpaces to specific scenes (default: all 18)')
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

    # Audio / spectrogram
    parser.add_argument('--sample-max-sec', type=float, default=0.6, help='Max RIR length in seconds')
    parser.add_argument('--nSamples', type=int, default=None, help='Cap dataset size (None = all)')
    parser.add_argument('--hop-length', type=int, default=64)
    parser.add_argument('--n-fft', type=int, default=256)
    parser.add_argument('--sr-target', type=int, default=22050, help='Target sampling rate')
    parser.add_argument('--scale-rir', type=str2bool, default=True)
    parser.add_argument('--apply-zero-tail', type=str2bool, default=False, help='Zero RIR values after -40 dB (requires --scale-rir)')
    parser.add_argument('--db-cutoff', type=float, default=-40.0, help='dB threshold for EDC-based RIR scaling and tail zeroing')

    # Data split
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--eval-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--split-by-room', type=str2bool, default=False, help='Split by room ID to avoid data leakage (GTU only)')
    parser.add_argument('--random-seed', type=int, default=42)

    # Post-training evaluation
    parser.add_argument('--post-train-eval', type=str2bool, default=True, help='Run full_model_eval.py on the finished run directory after training completes')

    args = parser.parse_args()
    if args.data_path is None:
        args.data_path = get_datasets_folder() if args.dataset_name == 'gtu' else _DEFAULT_SS_ROOT
    return args

def load_model_config(config_path, dataset_name, use_rt60_condition=False):
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
    # input_cond_dim: base is 9 (SoundSpaces) or 10 (GTU); each active flag adds 1
    src_trgt_dist_cond = model_config.pop('src_trgt_dist_cond')
    if dataset_name == 'soundspaces':
        base_dim = 10 if use_rt60_condition else 9
    else:
        base_dim = 10
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

def get_datasets_folder():
    """
    Returns the path to the dataset folder based on the platform (local PC or server).
    """
    if platform.system() == "Windows": # my local PC
        return os.path.normpath('C:/Yuval/MSc/AV_RIR/code_exp/rir_encoder/data/GTU_RIR_1024samples.pickle.dat')
    else: # on the server
         return os.path.normpath('./datasets/GTU_rir/GTU_RIR.pickle.dat')


def make_train_dataloader(dataset, args, collate_fn):
    """Build the training DataLoader, using EpochSubsetSampler when nSamples is set."""
    if args.dataset_name == 'soundspaces' and args.nSamples and args.nSamples < len(dataset):
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
    model_config, ie_config, src_trgt_dist_cond, loss_weighting = load_model_config(args.model_config, args.dataset_name, args.use_rt60_condition)

    # ---------- Load datasets ----------
    print("\n----------- Loading datasets... -----------\n")
    if args.dataset_name == 'gtu':
        train_dataset, eval_dataset, test_dataset = load_rir_dataset(
            name='gtu', path=args.data_path, split=True, mode='raw',
            hop_length=args.hop_length, n_fft=args.n_fft, use_spectrogram=True,
            sample_max_sec=args.sample_max_sec, nSamples=args.nSamples, sr_target=args.sr_target,
            train_ratio=args.train_ratio, eval_ratio=args.eval_ratio, test_ratio=args.test_ratio,
            random_seed=args.random_seed, split_by_room=args.split_by_room,
        )
    else:  # soundspaces
        rir_view_type      = None if args.rir_view_type      == 'none' else args.rir_view_type
        room_overview_type = None if args.room_overview_type == 'none' else args.room_overview_type
        image_size = ie_config.get('image_size') if ie_config else None
        train_dataset, eval_dataset, test_dataset = load_rir_dataset(
            name='soundspaces', rir_root=args.data_path, image_root=args.image_root,
            rir_view_type=rir_view_type, image_size=image_size,
            room_overview_type=room_overview_type,
            room_overview_config=args.room_overview_config,
            sample_max_sec=args.sample_max_sec, sr_target=args.sr_target,
            scenes=args.scenes, train_ratio=args.train_ratio, eval_ratio=args.eval_ratio,
            test_ratio=args.test_ratio, random_seed=args.random_seed, use_rt60=args.use_rt60_condition,
        )

    print(f"Dataset splits: {len(train_dataset)} - {len(eval_dataset)} - {len(test_dataset)}")

    # ---------- Create dataloaders ----------
    collate_fn = scale_and_spectrogram_collate_fn(
        sr=args.sr_target, db_cutoff=args.db_cutoff, n_fft=args.n_fft, hop_length=args.hop_length,
        scale_rir_flag=args.scale_rir, use_spectrogram=False, apply_zero_tail=args.apply_zero_tail,
        dataset_type=args.dataset_name,
    )
    
    train_dataloader = make_train_dataloader(train_dataset, args, collate_fn)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, drop_last=False, pin_memory=torch.cuda.is_available())

    sample_size = get_sample_size(train_dataloader, args.n_fft, args.hop_length)
    run_header = build_run_header(args, ie_config, git_hash, model_config, sample_size)
    print(f"\nxxxxxxx\nDataloader splits: {len(train_dataloader)} - {len(eval_dataloader)}\nxxxxxxx\n")


    # ---------- Model ----------
    # Build ImageEncoder when image_root is provided for SoundSpaces
    image_encoder = None
    if args.dataset_name == 'soundspaces' and args.image_root is not None:
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

        save_run_config(model_path, args, model.config, ie_config, git_hash)

        with open(os.path.join(configs_dir, 'model_config.json'), 'w') as f:
            json.dump(model.config, f, indent=2)

        import shutil
        shutil.copy(args.model_config, os.path.join(configs_dir, 'model_config_original.json'))

        if args.dataset_name == 'soundspaces' and args.room_overview_config is not None:
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