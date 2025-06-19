# train.py (rewritten to match FiLM-conditioned DiffusionModel setup)
import os
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
from RIRDiffusionModel import RIRDiffusionModel
from diffusers import DDPMScheduler
from trainer import DiffusionTrainer


# CUDA_VISIBLE_DEVICES=0,1,2,3 /home/yuvalmad/python312/bin/accelerate launch --multi_gpu --num_processes=4 ./Projects/rir_encoder/rir_generator/hugging_face/train_hf.py

def get_datasets_folder():
    """
    Returns the path to the dataset folder based on the platform (local PC or server).
    """
    if platform.system() == "Windows": # my local PC
        return os.path.normpath('C:/Yuval/MSc/AV_RIR/code_exp/rir_encoder/data/GTU_RIR_1024samples.pickle.dat')
    else: # on the server
         return os.path.normpath('./datasets/GTU_rir/GTU_RIR.pickle.dat')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default=get_datasets_folder())
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hop-length', type=int, default=128)
    parser.add_argument('--n-fft', type=int, default=128)
    parser.add_argument('--n-timesteps', type=int, default=512)
    parser.add_argument('--light-mode', type=bool, default=False)
    parser.add_argument('--checkpoint-freq', type=int, default=100)
    parser.add_argument('--use-cond-encoder', type=bool, default=True)

    return parser.parse_args()

def get_sample_size(dataloader):
    for batch in dataloader:
        rir = batch[0]
        return rir.shape[2:] if rir.ndim >= 3 else rir.shape[1:]

# ------------------------- Main --------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset
    dataset = load_rir_dataset('gtu', args.dataset_path, mode='raw',hop_length=args.hop_length, n_fft=args.n_fft, use_spectrogram=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    sample_size = get_sample_size(dataloader)
    data_info = {"n_fft": args.n_fft, "hop_length": args.hop_length, "use_spectrogram": True, "sample_size": sample_size}

    # Model
    model = RIRDiffusionModel(device=device, 
                 sample_size=sample_size, 
                 n_timesteps=args.n_timesteps, 
                 use_cond_encoder=args.use_cond_encoder,
                 light_mode=args.light_mode)
    # Scheduler 
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.n_timesteps)
    # Optimizer 
    optimizer = torch.optim.AdamW(model.get_model_params(), lr=args.lr)
    # Accelerator
    accelerator = Accelerator()
    # Trainer
    trainer = DiffusionTrainer(device=device,
                               model=model,
                               noise_scheduler=noise_scheduler,
                               optimizer=optimizer,
                               accelerator=accelerator,
                               lr = args.lr,
                               epochs=args.epochs,
                               checkpoint_freq=args.checkpoint_freq,
                               data_info=data_info,
                               )
    trainer.train(dataloader=dataloader)
    print('finished training!')

if __name__ == '__main__':
    main()
