import os
import time
import datetime
import logging
import torch
import torch.nn.functional as F
import math
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
                 checkpoint_freq=10,
                 eval_freq=5,
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
        self.use_amp = torch.cuda.is_available() and self.device.type == 'cuda'
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
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

    
    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None):
        """
        Train the diffusion model with optional evaluation.
        """
        # ============= train setup =============
        self.model.train()
        scaler = GradScaler() if self.use_amp else None
        losses_per_epoch = {'train_loss': [], 'train_norm_loss': [], 'eval_loss': [], 'eval_norm_loss': []}
        cur_epoch_losses = {'train_loss': 0, 'train_norm_loss': 0,'eval_loss': 0,'eval_norm_loss': 0}
        best_loss = float('inf')
        train_start_time = time.time()
        best_model_dict = None
        self.train_dataloader_len = len(train_dataloader)
        self.eval_dataloader_len = len(eval_dataloader)
        new_logdir = os.path.join(os.path.dirname(os.path.dirname(self.logdir)), 'finished', os.path.basename(self.logdir))
        
        # ============= Prepare with accelerator =============
        self.model, self.optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader)

        if self.accelerator.is_main_process:
            logging.info(f"*** Start RIR-GEN Diffusion Training ***\n"
                        f"          [Accelerator] is_distributed: {self.accelerator.distributed_type != 'NO'} | nProcesses: {self.accelerator.num_processes} | Device: {self.accelerator.device}\n"
                        f"          [Dataloader] Train size: {len(train_dataloader.dataset)} | len(train_loader): {len(train_dataloader)} | Val size: {len(eval_dataloader.dataset)}\n"
                        f"          [RunParams] Epochs: {self.epochs} | Batch size: {math.ceil(len(train_dataloader.dataset) / len(train_dataloader))} | Eval freq: {self.eval_freq}\n"
                        f"          [Model] LR: {self.lr} | Sample-Size: {self.data_info["sample_size"]} | n_timesteps: {self.model.n_timesteps}\n" #  | nParams: {self.model.count_parameters()}
                        f"          [Data] {self.data_info}\n\n")
            
        # ============= Start Epoch loop =============
        for epoch in range(self.epochs):
            # ---------- TRAINING ----------
            cur_epoch_losses['train_loss'], cur_epoch_losses['train_norm_loss'] = self._training_epoch(train_dataloader, epoch, scaler)

            # ---------- EVALUATION ----------
            cur_epoch_losses['eval_loss'], cur_epoch_losses['eval_norm_loss']= self._evaluation_epoch(eval_dataloader, epoch)

            # ---------- EPOCH WRAP-UP ----------            
            if self.accelerator.is_main_process:
                best_loss, best_model_dict, losses_per_epoch = self._epoch_wrapup(
                    epoch, cur_epoch_losses, best_loss, losses_per_epoch, 
                    train_start_time, best_model_dict, lr=self.optimizer.param_groups[0]['lr'])

        # ============= Training's finish =============
        # save Best model
        if self.accelerator.is_main_process:
            torch.save(best_model_dict, os.path.join(self.logdir, f'model_best.pth.tar'))
            
            # save metrics plots
            self._plot_all_metrics(losses_per_epoch)
            
            # move run outputs to the finished folder
            shutil.move(self.logdir, new_logdir)
            
            # final log
            total_elapsed_time = datetime.timedelta(seconds=int(time.time() - train_start_time))
            final_msg = f"""###########\nTraining finished successfully after {total_elapsed_time}
                         Best model saved at epoch {best_model_dict['epoch']} with train loss {best_loss:.4f}\n
                         save path: {new_logdir}\n###########"""
            logging.info(final_msg)
            print(final_msg)
            
        return new_logdir

    def _training_epoch(self, train_dataloader, epoch, scaler):
        """Complete training epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_norm_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")

        for rir, room_dim, mic_loc, speaker_loc, rt60 in progress_bar:
            loss_value, norm_loss_value = self._forward_step(rir, room_dim, mic_loc, speaker_loc, rt60, training=True, scaler=scaler)
            
            if self.accelerator.is_main_process:
                epoch_loss += loss_value
                epoch_norm_loss += norm_loss_value
        
        return epoch_loss, epoch_norm_loss

    def _evaluation_epoch(self, eval_dataloader, epoch):
        """Complete evaluation epoch"""
        if (epoch + 1) % self.eval_freq == 0:
            epoch_loss = 0.0
            epoch_norm_loss = 0.0
            self.model.eval()
            progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Eval]")
            
            with torch.no_grad():
                for rir, room_dim, mic_loc, speaker_loc, rt60 in progress_bar:
                    loss_value, norm_loss_value = self._forward_step(rir, room_dim, mic_loc, speaker_loc, rt60, training=False)
                    
                    if self.accelerator.is_main_process:
                        epoch_loss += loss_value
                        epoch_norm_loss += norm_loss_value
        else:
            epoch_loss = None
            epoch_norm_loss = None

        
        return epoch_loss, epoch_norm_loss

    def _forward_step(self, rir, room_dim, mic_loc, speaker_loc, rt60, training=True, scaler=None):
        """Common forward step for training and evaluation"""
        # prepare data
        rir = rir.to(self.device) # [B, 1, T] or [B, F, T]  
        condition = torch.cat([room_dim, mic_loc, speaker_loc, rt60.unsqueeze(1)], dim=1).to(self.device).float() # [B, 10]
        noise = torch.randn_like(rir).to(self.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (rir.shape[0],),
                                  device=self.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_rirs = self.noise_scheduler.add_noise(rir, noise, timesteps)

        if training:
            self.optimizer.zero_grad()

        # Predict the noise residual and Compute loss
        if self.use_amp:
            with autocast("cuda"):
                prediction = self.model(noisy_rirs, timesteps, condition.unsqueeze(1))
                noise_pred = prediction["sample"]
                loss = F.mse_loss(noise_pred, noise)
            
            if training:
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
        else:
            prediction = self.model(noisy_rirs, timesteps, condition.unsqueeze(1))
            noise_pred = prediction["sample"]
            loss = F.mse_loss(noise_pred, noise)
            
            if training:
                loss.backward()
                self.optimizer.step()
            
        # gather loss values across all processes (for logging)
        loss_value = self.accelerator.gather_for_metrics(loss.detach()).mean().item()
        norm_loss_value = self.calculate_norm_loss(noisy_rirs, loss).mean().item()
        
        return loss_value, norm_loss_value

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

    def _plot_all_metrics(self, losses_per_epoch):
        """Save all metrics plots"""
        save_metric(losses_per_epoch['train_loss'], 'log-train-loss', self.logdir, apply_log=True)
        save_metric(losses_per_epoch['train_loss'], 'train-loss', self.logdir, apply_log=False)
        save_metric(losses_per_epoch['train_norm_loss'], 'log-train-norm-loss', self.logdir, apply_log=True)
        save_metric(losses_per_epoch['train_norm_loss'], 'train-norm-loss', self.logdir, apply_log=False)
        
        save_metric(losses_per_epoch['eval_loss'], 'log-eval-loss', self.logdir, apply_log=True)
        save_metric(losses_per_epoch['eval_loss'], 'eval-loss', self.logdir, apply_log=False)
        save_metric(losses_per_epoch['eval_norm_loss'], 'log-eval-norm-loss', self.logdir, apply_log=True)
        save_metric(losses_per_epoch['eval_norm_loss'], 'eval-norm-loss', self.logdir, apply_log=False)

    def _epoch_wrapup(self, epoch, cur_epoch_losses, best_loss, losses_per_epoch, 
                     train_start_time, best_model_dict, lr):
        """
        Wrap up the epoch, check for best model, save checkpoints, and log metrics.
        """
        # Calculate average losses
        avg_train_loss = cur_epoch_losses['train_loss'] / self.train_dataloader_len
        avg_train_norm_loss = cur_epoch_losses['train_norm_loss'] / self.train_dataloader_len
        avg_eval_loss = cur_epoch_losses['eval_loss'] / self.eval_dataloader_len if cur_epoch_losses['eval_loss'] is not None else None
        avg_eval_norm_loss = cur_epoch_losses['eval_norm_loss'] / self.eval_dataloader_len if cur_epoch_losses['eval_norm_loss'] is not None else None
        
        # Append to epoch histories
        losses_per_epoch['train_loss'].append(avg_train_loss)
        losses_per_epoch['train_norm_loss'].append(avg_train_norm_loss)
        losses_per_epoch['eval_loss'].append(avg_eval_loss)
        losses_per_epoch['eval_norm_loss'].append(avg_eval_norm_loss)

        # Check for best model 
        if avg_eval_loss is not None and avg_eval_loss < best_loss:
            # Save the new best model
            best_loss = avg_eval_loss     
            model_unwrapped = self.accelerator.unwrap_model(self.model)
            best_model_dict = {
                'epoch': epoch + 1,
                'use_cond_encoder': model_unwrapped.use_cond_encoder,
                'model_nParams': model_unwrapped.count_parameters(),
                'model_config': model_unwrapped.config,
                'sample_size': self.data_info.get('sample_size', None),
                'data_info': self.data_info,
                'state_dict': model_unwrapped.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses_per_epoch': losses_per_epoch
            }
            
        # Save best model every checkpoint_freq epochs
        if min(epoch, self.checkpoint_freq) > 0 and epoch % self.checkpoint_freq == 0:
            torch.save(best_model_dict, os.path.join(self.logdir, f'model_best.pth.tar'))

        # Log epoch results
        total_elapsed_time = datetime.timedelta(seconds=int(time.time() - train_start_time))
        eval_loss_str = f"{avg_eval_loss:.4f}" if avg_eval_loss is not None else "N/A"
        logging.info(f"[Epoch {epoch+1}] Elapsed: {str(total_elapsed_time)}\n"
                    f"         Train Loss: {avg_train_loss:.4f} | Eval Loss: {eval_loss_str} | LR: {lr:.6f}")
            
        return best_loss, best_model_dict, losses_per_epoch