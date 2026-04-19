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
from utils.misc import str2bool

_DEFAULT_CFG     = '/home/yuvalmad/Projects/Gen-RIR-Diffusion/config/model_config_cond_encoder_cfg.json'
_DEFAULT_SS_ROOT = '/dsi/gannot-lab/gannot-lab1/datasets/SoundSpaces/binaural_rirs/replica/'

"""
# GTU (original):
CUDA_VISIBLE_DEVICES=3 python3 ./Projects/Gen-RIR-Diffusion/run_train.py --batch-size 16 --epochs 100 --nSamples 128 \
|& tee -a "./Projects/Gen-RIR-Diffusion/outputs/logs/train_$(hostname -s)_$(date +%F_%H-%M-%S).log"

# SoundSpaces — scalar conditioning only (no images):
CUDA_VISIBLE_DEVICES=3 python3 ./Projects/Gen-RIR-Diffusion/run_train.py --dataset-name soundspaces --batch-size 4 --epochs 50 \
|& tee -a "./Projects/Gen-RIR-Diffusion/outputs/logs/train_$(hostname -s)_$(date +%F_%H-%M-%S).log"

# SoundSpaces — with image conditioning (model_config_VisualCond.json):
CUDA_VISIBLE_DEVICES=3 python3 ./Projects/Gen-RIR-Diffusion/run_train.py --dataset-name soundspaces --batch-size 4 --epochs 50 \
    --model-config ./Projects/Gen-RIR-Diffusion/config/model_config_VisualCond.json \
    --image-root /dsi/gannot-lab/gannot-lab1/datasets/Replica_rendered/target2source_rgb/ \
|& tee -a "./Projects/Gen-RIR-Diffusion/outputs/logs/train_$(hostname -s)_$(date +%F_%H-%M-%S).log"

# SoundSpaces — initial test on a few scenes (no images):
CUDA_VISIBLE_DEVICES=3 python3 ./Projects/Gen-RIR-Diffusion/run_train.py --dataset-name soundspaces --batch-size 4 --epochs 10 \
    --scenes office_2 room_0 apartment_0 \
|& tee -a "./Projects/Gen-RIR-Diffusion/outputs/logs/train_$(hostname -s)_$(date +%F_%H-%M-%S).log"
"""


# ------------------------- Utils --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # Model configuration
    parser.add_argument('--model-config', type=str, default=_DEFAULT_CFG,
                        help='Path to model config JSON. Use model_config_VisualCond.json for image conditioning.')

    # Dataset selection
    parser.add_argument('--dataset-name', type=str, default='gtu', choices=['gtu', 'soundspaces'],
                        help='Dataset to train on. Defaults: gtu→GTU pickle, soundspaces→server RIR root')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Dataset path. Defaults to GTU pickle or SoundSpaces RIR root based on --dataset-name')
    parser.add_argument('--image-root', type=str, default=None,
                        help='Rendered RGB image root (target2source_rgb/); omit to disable image conditioning')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                        help='Restrict SoundSpaces to specific scenes (default: all 18)')

    # Training configuration
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n-timesteps', type=int, default=1000)
    parser.add_argument('--checkpoint-freq', type=int, default=5)
    parser.add_argument('--eval-freq', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--use-eval', type=str2bool, default=True, help='Use evaluation dataset during training')

    # Audio / spectrogram
    parser.add_argument('--sample-max-sec', type=int, default=1, help='Max RIR length in seconds')
    parser.add_argument('--nSamples', type=int, default=None, help='Cap dataset size (None = all)')
    parser.add_argument('--hop-length', type=int, default=64)
    parser.add_argument('--n-fft', type=int, default=256)
    parser.add_argument('--sr-target', type=int, default=22050, help='Target sampling rate')
    parser.add_argument('--scale-rir', type=str2bool, default=True)
    parser.add_argument('--apply-zero-tail', type=str2bool, default=False,
                        help='Zero RIR values after -40 dB (requires --scale-rir)')
    parser.add_argument('--db-cutoff', type=float, default=-40.0, help='dB cutoff for EDC cropping')

    # Data split
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--eval-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--split-by-room', type=str2bool, default=False,
                        help='Split by room ID to avoid data leakage (GTU only)')
    parser.add_argument('--random-seed', type=int, default=42)

    args = parser.parse_args()
    if args.data_path is None:
        args.data_path = get_datasets_folder() if args.dataset_name == 'gtu' else _DEFAULT_SS_ROOT
    return args

def load_model_config(config_path):
    """Load model configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

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

def gather_data_info(args, train_dataloader):
    sample_size = get_sample_size(train_dataloader, args.n_fft, args.hop_length)
    data_info = {"n_fft": args.n_fft,
                 "hop_length": args.hop_length, 
                 "use_spectrogram": True, 
                 "sample_size": sample_size,
                 "sample_max_sec": args.sample_max_sec, 
                 "sr_target": args.sr_target,
                 "nSamples": args.nSamples,
                 "db_cutoff": args.db_cutoff,
                 "scale_rir": args.scale_rir,
                 "apply_zero_tail": args.apply_zero_tail,
                 "train_ratio": args.train_ratio,
                 "eval_ratio": args.eval_ratio,
                 "test_ratio": args.test_ratio,
                 "split_by_room": args.split_by_room,
                 "random_seed": args.random_seed
                 }
    return data_info   

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

# ------------------------- Main --------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- debug mode ----------
    debug_mode = False
    if debug_mode:
        print("\nXXXXXXX\n Running in DEBUG mode!!!! \nXXXXXXX\n")
        args.batch_size = 2
        args.epochs = 2
        args.nSamples = 14
        args.split_by_room = False
        
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
        train_dataset, eval_dataset, test_dataset = load_rir_dataset(
            name='soundspaces', rir_root=args.data_path, image_root=args.image_root,
            scenes=args.scenes, train_ratio=args.train_ratio, eval_ratio=args.eval_ratio,
            test_ratio=args.test_ratio, random_seed=args.random_seed,
        )

    print(f"Dataset splits: {len(train_dataset)} - {len(eval_dataset)} - {len(test_dataset)}")

    # ---------- Create dataloaders ----------
    collate_fn = scale_and_spectrogram_collate_fn(
        sr=args.sr_target, db_cutoff=args.db_cutoff, n_fft=args.n_fft, hop_length=args.hop_length,
        scale_rir_flag=args.scale_rir, use_spectrogram=False, apply_zero_tail=args.apply_zero_tail,
        dataset_type=args.dataset_name,
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, drop_last=True, pin_memory=torch.cuda.is_available())

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, drop_last=False, pin_memory=torch.cuda.is_available())
    
    data_info = gather_data_info(args, train_dataloader)
    print(f"\nxxxxxxx\nDataloader splits: {len(train_dataloader)} - {len(eval_dataloader)}\nxxxxxxx\n")


    # ---------- Model ----------
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)

    # input_cond_dim is always derived from dataset: SoundSpaces has no RT60 (9-dim), GTU has RT60 (10-dim)
    model_config['input_cond_dim'] = 9 if args.dataset_name == 'soundspaces' else 10

    # Image encoder architecture config lives in JSON; pop it before passing **model_config to the model
    ie_config = model_config.pop('image_encoder_config', None)

    # Build ImageEncoder when image_root is provided for SoundSpaces
    image_encoder = None
    if args.dataset_name == 'soundspaces' and args.image_root is not None:
        assert ie_config is not None, "image_encoder_config missing from model config JSON"
        cross_attention_dim = model_config['block_out_channels'][-1]
        ie_config['image_size'] = tuple(ie_config['image_size'])  # JSON gives list, encoder expects tuple
        image_encoder = ImageEncoder(out_dim=cross_attention_dim, **ie_config)
        print(f"\nImageEncoder: DA3 ViT-{ie_config['model_variant']}, out_dim={cross_attention_dim}, "
              f"~{image_encoder.n_tokens(n_img=1)} tokens per sample\n")

    model = RIRDiffusionModel(
        device=device,
        sample_size=data_info['sample_size'],
        n_timesteps=args.n_timesteps,
        image_encoder=image_encoder,
        **model_config,
    )
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
        data_info=data_info,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    
    # ---------- Train ----------
    print("\n---------- Starting training... ----------\n")
    model_path = trainer.train(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)

    # ---------- Save model and training arguments ----------
    if accelerator.is_main_process:
        # Save training arguments and model configuration
        torch.save(args, os.path.join(model_path, 'run_args.pth'))

        # Save the model configuration used for this training
        model_config_path = os.path.join(model_path, 'model_config.json')
        with open(model_config_path, 'w') as f:
            json.dump(model.config, f, indent=2)
        
        print('\n---------- Training finished successfully!! ----------\n')

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