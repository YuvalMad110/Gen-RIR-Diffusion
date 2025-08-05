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
from data.dataset_collate_fn import scale_and_spectrogram_collate_fn
from RIRDiffusionModel import RIRDiffusionModel
from diffusers import DDPMScheduler
from trainer import DiffusionTrainer
from utils.signal_scaling2 import scaled_rir_collate_fn

# CUDA_VISIBLE_DEVICES=0,1,2,3 /home/yuvalmad/python312/bin/accelerate launch --multi_gpu --num_processes=4 ./Projects/Gen-RIR-Diffusion/run_train.py --batch-size 16 --epochs 100 --nSamples 128

def get_datasets_folder():
    """
    Returns the path to the dataset folder based on the platform (local PC or server).
    """
    if platform.system() == "Windows": # my local PC
        return os.path.normpath('C:/Yuval/MSc/AV_RIR/code_exp/rir_encoder/data/GTU_RIR_1024samples.pickle.dat')
    else: # on the server
         return os.path.normpath('./datasets/GTU_rir/GTU_RIR.pickle.dat')

def gather_data_info(args, dataloader):
    sample_size = get_sample_size(dataloader)
    data_info = {"n_fft": args.n_fft,
                 "hop_length": args.hop_length, 
                 "use_spectrogram": True, 
                 "sample_size": sample_size,
                 "sample_max_sec": args.sample_max_sec, 
                 "sr_target": args.sr_target,
                 "nSamples": args.nSamples,
                 "db_cutoff": args.db_cutoff,
                 "scale_rir": args.scale_rir,
                 "apply_zero_tail": args.apply_zero_tail
                 }
    return data_info
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default=get_datasets_folder())
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n-timesteps', type=int, default=1000)
    parser.add_argument('--light-mode', type=bool, default=False)
    parser.add_argument('--checkpoint-freq', type=int, default=5)
    parser.add_argument('--use-cond-encoder', type=bool, default=False)
    # Dataset specific arguments
    parser.add_argument('--sample-max-sec', type=int, default=1, help="None for no limit, otherwise in samples")
    parser.add_argument('--nSamples', type=int, default=None, help="Number of samples to use from the dataset, None for all")
    parser.add_argument('--hop-length', type=int, default=64)
    parser.add_argument('--n-fft', type=int, default=256)
    parser.add_argument('--sr-target', type=int, default=22050, help="Target sampling rate for the RIRs, if None, use original sampling rate")
    parser.add_argument('--scale-rir', type=bool, default=True)
    parser.add_argument('--apply-zero-tail', type=bool, default=True, help="Will zero all values of the RIR after -40db (only when scale_rir==True)")
    parser.add_argument('--db-cutoff', type=float, default=-40.0, help='dB cutoff for EDC cropping in rir scaling')

    return parser.parse_args()

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
    

def get_sample_size(dataloader):
    for batch in dataloader:
        rir = batch[0]
        return rir.shape[2:] if rir.ndim >= 3 else rir.shape[1:]

# ------------------------- Main --------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- dataset ----------
    dataset = load_rir_dataset('gtu', args.dataset_path, mode='raw',hop_length=args.hop_length, n_fft=args.n_fft,
                                use_spectrogram=True, sample_max_sec=args.sample_max_sec, nSamples=args.nSamples, sr_target=args.sr_target)
    
    collate_fn = scale_and_spectrogram_collate_fn(sr=args.sr_target, db_cutoff=args.db_cutoff, n_fft=args.n_fft, hop_length=args.hop_length, 
                                                  scale_rir_flag=args.scale_rir, use_spectrogram=True, apply_zero_tail=args.apply_zero_tail)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    data_info = gather_data_info(args, dataloader)
    # num_workers_test(dataset=dataset, batch_size=args.batch_size)    
    # return

    # ---------- Model ----------
    model = RIRDiffusionModel(device=device, 
                 sample_size=data_info['sample_size'], 
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
    # ---------- Train ----------
    trainer.train(dataloader=dataloader)
    print('finished training!')

if __name__ == '__main__':
    main()
