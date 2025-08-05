import os
import time
import datetime
import logging
import torch
import torch.nn.functional as F
import shutil
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from utils.misc import save_metric, get_timestamped_logdir, plot_signals
from tqdm import tqdm
from diffusers import DDPMScheduler

# ToDo:
#     - use UNet2DConditionModel with the block_out_channels=(32, 64, 128, 128) (duplicate first/last block from up_block_types and down_block_types)
#     - try transformer based model
#     - try different scheduler
#     - understand whether a conditioning encoder is necessary
#     - use RT60(freq)

class DiffusionTrainer():
    def __init__(self,
                 device, 
                 lr=1e-4, 
                 epochs=20, 
                 n_timesteps=1000, 
                 checkpoint_freq=10,
                 data_info=None,
                 model=None,
                 optimizer=None,
                 noise_scheduler=None,
                 accelerator=None,):
        """
        Initialize the RIR generator.
        """
        # -------- Cfg --------
        self.model = model
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.n_timesteps = n_timesteps
        self.use_amp = torch.cuda.is_available() and self.device.type == 'cuda'
        self.checkpoint_freq = checkpoint_freq
        self.logdir = get_timestamped_logdir('outputs/not_completed')
        self.data_info = data_info if data_info is not None else {}
         # Setup Accelerator for DDP/AMP
        self.accelerator = accelerator
        if self.accelerator.is_main_process:
            os.makedirs(self.logdir, exist_ok=False)
            logging.basicConfig(filename=os.path.join(self.logdir, "train.log"),
                                level=logging.DEBUG,
                                format='%(asctime)s - %(message)s')
            logging.getLogger('matplotlib').setLevel(logging.WARNING)

    
    def train(self, dataloader: DataLoader):
        """
        Train the diffusion model.
        """
        # -------- train setup --------
        self.model.train()
        scaler = GradScaler() if self.use_amp else None
        loss_per_epoch = []
        norm_loss_per_epoch = []
        best_loss = float('inf')
        train_start_time = time.time()
        best_model_dict = None
        # -------- Prepare with accelerator --------
        self.model, self.optimizer, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, dataloader)

        if self.accelerator.is_main_process:
            logging.info(f"*** Start RIR-GEN Diffusion Training ***\n"
                        f"          [Accelerator] is_distributed: {self.accelerator.distributed_type != 'NO'} | nProcesses: {self.accelerator.num_processes} | Device: {self.accelerator.device}\n"
                        f"          [Dataloader] Dataset size: {len(dataloader.dataset)} | len(train_loader): {len(dataloader)}\n"
                        f"          [RunParams] Epochs: {self.epochs} | Batch size: {dataloader.batch_size}\n"
                        f"          [Model] LR: {self.lr} | Sample-Size: {self.data_info["sample_size"]} | n_timesteps: {self.n_timesteps}\n"
                        f"          [Data] {self.data_info}\n\n")

        for epoch in range(self.epochs):
            # epoch setup
            epoch_loss = 0.0
            epoch_norm_loss  = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            # Bacth loop
            for rir, room_dim, mic_loc, speaker_loc, rt60 in progress_bar:
                # --------------------------------------- Forwad pass ---------------------------------------
                # prepare data
                rir = rir.to(self.device) # [B, 1, T]
                condition = torch.cat([room_dim, mic_loc, speaker_loc, rt60.unsqueeze(1)], dim=1).to(self.device).float() # [B, 10]
                noise = torch.randn_like(rir).to(self.device)

                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (rir.shape[0],),
                                          device=self.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_rirs = self.noise_scheduler.add_noise(rir, noise, timesteps)

                self.optimizer.zero_grad()
                # Predict the noise residual and Compute loss
                if self.use_amp:
                    with autocast("cuda"):
                        prediction = self.model(noisy_rirs, timesteps, condition.unsqueeze(1))
                        noise_pred = prediction["sample"]
                        loss = F.mse_loss(noise_pred, noise)
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    prediction = self.model(noisy_rirs, timesteps, condition.unsqueeze(1))
                    noise_pred = prediction["sample"]
                    loss = F.mse_loss(noise_pred, noise)
                    loss.backward()
                    self.optimizer.step()
                # gather loss values across all processes (for logging)
                loss_value = self.accelerator.gather_for_metrics(loss.detach()) # gather loss values across all processes
                norm_loss_value = self.calculate_norm_loss(noisy_rirs, loss)
                if self.accelerator.is_main_process:
                    epoch_loss += loss_value.mean().item() # batch accumulated loss, mean across processes
                    epoch_norm_loss += norm_loss_value.mean().item()
                # --------------------------------------- End Forward pass ---------------------------------------

             # ------------- End of Batch-loop -------------
            if self.accelerator.is_main_process:
                best_loss, best_model_dict, loss_per_epoch, norm_loss_per_epoch, progress_bar = self.epoch_wrapup(
                    epoch, epoch_loss, epoch_norm_loss, best_loss, loss_per_epoch, norm_loss_per_epoch, progress_bar, train_start_time, 
                    best_model_dict, dataloader_len=len(dataloader), lr=self.optimizer.param_groups[0]['lr'])

        # ------------- End of Epoch-loop -------------
        
        # save Best model
        if self.accelerator.is_main_process:
            torch.save(best_model_dict, os.path.join(self.logdir, f'model_best.pth.tar'))
            
             # save metrics plots
            save_metric(loss_per_epoch, 'log-loss', self.logdir, apply_log=True)
            save_metric(loss_per_epoch, 'loss', self.logdir, apply_log=False)
            save_metric(norm_loss_per_epoch, 'log-norm-loss', self.logdir, apply_log=True)
            save_metric(norm_loss_per_epoch, 'norm-loss', self.logdir, apply_log=False)
            # move run outputs to the finished folder
            new_logdir = os.path.join(os.path.dirname(os.path.dirname(self.logdir)), 'finished', os.path.basename(self.logdir))
            shutil.move(self.logdir, new_logdir)
            # final log
            total_elapsed_time = datetime.timedelta(seconds=int(time.time() - train_start_time))
            final_msg = f"""###########\nTraining finished successfully after {total_elapsed_time}
                         Best model saved at epoch {best_model_dict['epoch']} with loss {best_loss:.4f}\n
                         save path: {new_logdir}###########"""
            logging.info(final_msg)
            print(final_msg)
            


    def epoch_wrapup(self, epoch, epoch_loss, epoch_norm_loss, best_loss, loss_per_epoch, norm_loss_per_epoch, progress_bar, train_start_time, best_model_dict, dataloader_len, lr):
        """
        Wrap up the epoch, check for best model, save checkpoints, and log metrics.
        """
        avg_epoch_loss = epoch_loss / dataloader_len # average over batches and processes
        loss_per_epoch.append(avg_epoch_loss)
        avg_epoch_norm_loss = epoch_norm_loss / dataloader_len # average over batches and processes
        norm_loss_per_epoch.append(avg_epoch_norm_loss)

        # check for best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model_unwrapped = self.accelerator.unwrap_model(self.model)
            best_model_dict = {
                'epoch': epoch + 1,
                'use_cond_encoder': model_unwrapped.use_cond_encoder,
                'light_mode': model_unwrapped.light_mode,
                'model_nParams': model_unwrapped.count_parameters(),
                'n_timesteps': self.n_timesteps,
                'sample_size': self.data_info.get('sample_size', None),
                'data_info': self.data_info,
                'state_dict': model_unwrapped.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss_per_epoch': loss_per_epoch,
                'norm_loss_per_epoch': norm_loss_per_epoch
            }
            
        # Save best model every checkpoint_freq epochs
        if min(epoch, self.checkpoint_freq) > 0 and epoch % self.checkpoint_freq == 0:
            torch.save(best_model_dict, os.path.join(self.logdir, f'model_best.pth.tar'))

        # log ETA and epoch loss
        eta = ((self.epochs - epoch - 1) * dataloader_len + dataloader_len - progress_bar.n) / (progress_bar.format_dict["rate"] or 1)
        eta_td = datetime.timedelta(seconds=int(eta))
        total_elapsed_time = datetime.timedelta(seconds=int(time.time() - train_start_time))
        logging.info(f"[Epoch {epoch+1}] Elapsed: {str(total_elapsed_time)} | ETA: {eta_td}\n"
                    f"         Loss: {avg_epoch_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        progress_bar.set_postfix({'epoch': epoch + 1, 'loss': avg_epoch_loss,
                    'lr': lr, 'eta': str(eta_td)})
            
        return best_loss, best_model_dict, loss_per_epoch, norm_loss_per_epoch, progress_bar


    # @torch.no_grad()
    # def generate(self, cond: torch.Tensor, length: int = 2048, num_steps: int = 50):
    #     self.model.eval()
    #     noise = torch.randn((1, 1, length), device=self.device)
    #     context = self.condition_encoder(cond.to(self.device).unsqueeze(0))
    #     return self.model.sample(noise, features=context, num_steps=num_steps)

    @torch.no_grad()
    def generate(self, cond: torch.Tensor, shape: torch.Size, num_steps: int = 50):
        """
        Generate a synthetic RIR conditioned on input parameters.

        Args:
            cond (torch.Tensor): Tensor of shape [10] or [B, 10] containing the conditioning parameters.
            length (int): The desired RIR signal length.
            num_steps (int): Number of reverse diffusion steps (can be less than n_timesteps for faster sampling).

        Returns:
            torch.Tensor: Generated RIR signal of shape [1, 1, length]
        """
        self.model.eval()
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = cond.float().to(self.device)  # [1, 10]

        # Use encoder if defined
        if self.model.use_cond_encoder:
            cond = self.model.condition_encoder(cond)  # [1, 128]
        cond = cond.unsqueeze(1)  # [B, 1, C] for model

        # Initialize with Gaussian noise
        noisy_rir = torch.randn(shape, device=self.device)

        # Set up inference timesteps
        inference_scheduler = DDPMScheduler(num_train_timesteps=self.n_timesteps)
        inference_scheduler.set_timesteps(num_steps)
        
        for t in inference_scheduler.timesteps:
            model_output = self.model(noisy_rir, t, cond)["sample"]
            noisy_rir = inference_scheduler.step(model_output, t, noisy_rir).prev_sample

        return noisy_rir.cpu()

    def calculate_norm_loss(self, noisy_rirs, loss):
        """
        Calculate the normalized loss between the predicted noise and the true noise.
        For display alone (not used for model training)
        """
        B = noisy_rirs.shape[0]
        norm_factor = noisy_rirs.view(B, -1).std(dim=1, keepdim=True).view(B, 1, 1, 1)

        norm_loss = loss.detach() / (norm_factor ** 2)
        norm_loss_value = self.accelerator.gather_for_metrics(norm_loss.detach())
        return norm_loss_value
    
