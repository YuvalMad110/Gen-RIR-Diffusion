import os
import math
import time
import datetime
import logging
import torch
import torch.nn.functional as F
import shutil
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from utils.misc import save_metric, get_timestamped_logdir, plot_signals
from utils.signal_proc import batch_c50, batch_edc_error, spectrogram_to_waveform, unscale_rir, waveform_to_spectrogram
from utils.dataset_utils import build_condition_tensor
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
                 model=None,
                 optimizer=None,
                 noise_scheduler=None,
                 accelerator=None,
                 n_fft=256,
                 hop_length=64,
                 run_header="",
                 src_trgt_dist_cond=False,
                 sr=None,
                 loss_weighting=None,
                 c50_loss=None,
                 edc_loss=None,
                 scale_rir=False,
                 patience=15):
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
        self.patience = patience   # stop after this many epochs without a new best eval loss
        self.logdir = get_timestamped_logdir('outputs/not_completed')
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.run_header = run_header
        self.src_trgt_dist_cond = src_trgt_dist_cond
        self.sr = sr
        self.loss_weighting = loss_weighting
        self.c50_loss = c50_loss
        self.edc_loss = edc_loss
        self.scale_rir = scale_rir
        # Setup Accelerator for DDP/AMP
        self.accelerator = accelerator
        if self.accelerator.is_main_process:
            os.makedirs(self.logdir, exist_ok=False)
            os.makedirs(os.path.join(self.logdir, 'plots'), exist_ok=False)
            os.makedirs(os.path.join(self.logdir, 'configs'), exist_ok=False)
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
            self._log_run_header(train_dataloader, eval_dataloader)

        # ============= Start Epoch loop =============
        for epoch in range(self.epochs):
            # Advance EpochSubsetSampler seed so each epoch draws a different random subset
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)

            # ---------- TRAINING ----------
            cur_epoch_losses['train_loss'], cur_epoch_losses['train_norm_loss'] = self._training_epoch(train_dataloader, epoch, scaler)

            # ---------- EVALUATION ----------
            cur_epoch_losses['eval_loss'], cur_epoch_losses['eval_norm_loss']= self._evaluation_epoch(eval_dataloader, epoch)

            # ---------- EPOCH WRAP-UP ----------
            if self.accelerator.is_main_process:
                best_loss, best_model_dict, losses_per_epoch = self._epoch_wrapup(
                    epoch, cur_epoch_losses, best_loss, losses_per_epoch,
                    train_start_time, best_model_dict, lr=self.optimizer.param_groups[0]['lr'])

            # ---------- EARLY STOPPING ----------
            if self._should_stop_early(epoch, best_model_dict):
                break

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
                         Best model saved at epoch {best_model_dict['epoch']} with eval loss {best_loss:.4f}\n
                         save path: {new_logdir}\n###########"""
            logging.info(final_msg)
            print(final_msg)
            
        return new_logdir

    def _log_run_header(self, train_dataloader, eval_dataloader):
        _P = "          "
        logging.info(
            f"*** Start RIR-GEN Diffusion Training ***\n"
            f"{self.run_header}\n"
            f"{_P}[Accelerator] Distributed: {self.accelerator.distributed_type != 'NO'} | Processes: {self.accelerator.num_processes} | Device: {self.accelerator.device}\n"
            f"{_P}[Sizes]       Train: {len(train_dataloader.dataset)} ({len(train_dataloader)} batches) | Val: {len(eval_dataloader.dataset)}\n\n"
        )

    def _training_epoch(self, train_dataloader, epoch, scaler):
        """Complete training epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_norm_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")

        for step, batch in enumerate(progress_bar):
            loss_value, norm_loss_value = self._forward_step(batch, training=True, scaler=scaler)
            self._assert_finite_loss(loss_value, epoch, step, batch)
            if self.accelerator.is_main_process:
                epoch_loss      += loss_value
                epoch_norm_loss += norm_loss_value

        return epoch_loss, epoch_norm_loss

    def _assert_finite_loss(self, loss_value, epoch, step, batch):
        """Stop the run on a non-finite loss, which no later epoch can recover from."""
        if math.isfinite(loss_value):
            return
        msg = (f"Loss is {loss_value} at epoch {epoch+1}, step {step+1}. "
               f"Batch peak |rir| {batch['rir'].abs().amax().item():.3g}, "
               f"scenes {sorted(set(batch['scene']))[:4]}. "
               f"Values above 65504 overflow the mixed-precision forward pass.")
        logging.error(msg)
        raise RuntimeError(msg)

    def _evaluation_epoch(self, eval_dataloader, epoch):
        """Complete evaluation epoch"""
        if (epoch + 1) % self.eval_freq == 0:
            epoch_loss = 0.0
            epoch_norm_loss = 0.0
            self.model.eval()
            progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Eval]")

            with torch.no_grad():
                for batch in progress_bar:
                    loss_value, norm_loss_value = self._forward_step(batch, training=False)
                    if self.accelerator.is_main_process:
                        epoch_loss      += loss_value
                        epoch_norm_loss += norm_loss_value
        else:
            epoch_loss = epoch_norm_loss = None

        return epoch_loss, epoch_norm_loss

    def _compute_loss(self, noise_pred, noise, batch, noisy_rirs, timesteps):
        """The training objective: the noise loss, plus whichever of the clarity and decay-curve
        terms are configured.

        Args:
            noise_pred: Predicted noise [batch, 2, freq, frame]
            noise:      The noise that was added [batch, 2, freq, frame]
            batch:      The collated batch
            noisy_rirs: The spectrogram the prediction was made from [batch, 2, freq, frame]
            timesteps:  Diffusion timestep per sample [batch]

        Returns:
            The scalar the backward pass runs on.
        """
        loss = self._noise_loss(noise_pred, noise, batch)
        if self.c50_loss is not None:
            loss = loss + self.c50_loss['weight'] * self._c50_term(noise_pred, noisy_rirs, timesteps, batch)
        if self.edc_loss is not None:
            loss = loss + self.edc_loss['weight'] * self._edc_term(noise_pred, noisy_rirs, timesteps, batch)
        return loss

    def _denoised_waveforms(self, noise_pred, noisy_rirs, timesteps, batch):
        """The waveform the noise estimate implies and the batch's own, at their natural decay.

        The prediction is the clean spectrogram the model's noise estimate implies, brought back to
        the time domain and, when training scales the RIRs, returned to its natural decay. The
        reference is the batch's own waveform under the same treatment, so both sides are measured
        the same way. Both are cut to the length they share.

        Args:
            noise_pred: Predicted noise [batch, 2, freq, frame]
            noisy_rirs: The spectrogram the prediction was made from [batch, 2, freq, frame]
            timesteps:  Diffusion timestep per sample [batch]
            batch:      The collated batch, supplying 'rir' and, when scaling, 'rt60'

        Returns:
            (predicted, reference, alphas): the two waveforms [batch, time] and the scheduler's
            cumulative signal coefficient at each sample's timestep [batch].
        """
        device = noise_pred.device
        alphas = self.noise_scheduler.alphas_cumprod.to(device)[timesteps]         # [batch]

        shaped    = alphas.view(-1, 1, 1, 1)
        denoised  = (noisy_rirs.float() - (1 - shaped).sqrt() * noise_pred.float()) / shaped.sqrt()
        predicted = spectrogram_to_waveform(denoised, hop_length=self.hop_length, n_fft=self.n_fft)
        reference = batch['rir'].to(device).float()                                # [batch, time]

        length    = min(predicted.shape[-1], reference.shape[-1])
        predicted, reference = predicted[..., :length], reference[..., :length]

        if self.scale_rir:
            rt60      = batch['rt60'].to(device)
            predicted = unscale_rir(predicted, self.sr, rt60)
            reference = unscale_rir(reference, self.sr, rt60)

        return predicted, reference, alphas

    def _timestep_weight(self, alphas, min_snr_gamma, keep):
        """min(SNR(t), min_snr_gamma) over the kept samples.

        This is the factor that gives a quantity read off the clean signal the balance across
        timesteps that the noise loss has by construction.
        """
        return (alphas / (1 - alphas)).clamp(max=min_snr_gamma)[keep]

    def _c50_term(self, noise_pred, noisy_rirs, timesteps, batch):
        """Mean absolute clarity error between the predicted and the reference RIR, in dB.

        Args:
            noise_pred: Predicted noise [batch, 2, freq, frame]
            noisy_rirs: The spectrogram the prediction was made from [batch, 2, freq, frame]
            timesteps:  Diffusion timestep per sample [batch]
            batch:      The collated batch, supplying 'rir' and, when scaling, 'rt60'

        Returns:
            Weighted mean absolute clarity error in dB, a scalar. Zero when no sample in the batch
            carries a defined clarity on both sides.
        """
        predicted, reference, alphas = self._denoised_waveforms(noise_pred, noisy_rirs, timesteps, batch)

        eps = self.c50_loss['eps']
        predicted_c50, predicted_valid = batch_c50(predicted, self.sr, eps=eps)
        reference_c50, reference_valid = batch_c50(reference, self.sr, eps=eps)

        keep = predicted_valid & reference_valid
        if not bool(keep.any()):
            return noise_pred.new_zeros(())

        weight = self._timestep_weight(alphas, self.c50_loss['min_snr_gamma'], keep)
        error  = (predicted_c50 - reference_c50).abs()[keep]
        return (weight * error).sum() / weight.sum()

    def _edc_term(self, noise_pred, noisy_rirs, timesteps, batch):
        """Squared error between the predicted and the reference energy decay curve.

        Both curves start at their own signal's peak and are normalised by the energy of the window
        they share, so the term reads decay shape alone and carries no absolute gain. The error is
        in dB^2 or dimensionless according to the configured form.

        Args:
            noise_pred: Predicted noise [batch, 2, freq, frame]
            noisy_rirs: The spectrogram the prediction was made from [batch, 2, freq, frame]
            timesteps:  Diffusion timestep per sample [batch]
            batch:      The collated batch, supplying 'rir' and, when scaling, 'rt60'

        Returns:
            Weighted mean squared decay-curve error, a scalar. Zero when no sample in the batch
            carries a long enough shared window on both sides.
        """
        predicted, reference, alphas = self._denoised_waveforms(noise_pred, noisy_rirs, timesteps, batch)

        error, keep = batch_edc_error(predicted, reference, self.sr, form=self.edc_loss['form'],
                                      floor_db=self.edc_loss.get('floor_db'),
                                      min_post_peak_ms=self.edc_loss['min_post_peak_ms'],
                                      energy_floor=self.edc_loss['energy_floor'])
        if not bool(keep.any()):
            return noise_pred.new_zeros(())

        weight = self._timestep_weight(alphas, self.edc_loss['min_snr_gamma'], keep)
        return (weight * error[keep]).sum() / weight.sum()

    def _noise_loss(self, noise_pred, noise, batch):
        """Error between the predicted and the added noise, optionally weighted by a per-sample
        RIR decay mask."""
        if self.loss_weighting is None or self.loss_weighting['method'] == 'none':
            return F.mse_loss(noise_pred, noise)

        if self.loss_weighting['method'] == 'rt60_binary':
            rt60 = batch['rt60'].to(noise.device)                          # [B]
            alpha = abs(self.loss_weighting['db_threshold']) / 60.0
            T = noise.shape[-1]                                            # spectrogram time frames
            frame_rate = self.sr / self.hop_length
            t = torch.arange(T, device=noise.device) / frame_rate         # [T] seconds
            cutoff = (rt60 * alpha).clamp(max=T / frame_rate)             # [B]
            mask = (t.unsqueeze(0) < cutoff.unsqueeze(1)).float()         # [B, T]
            mask = mask[:, None, None, :]                                  # [B, 1, 1, T] broadcast over C, F
            sq_err = (noise_pred - noise) ** 2
            return (mask * sq_err).sum() / mask.expand_as(sq_err).sum()

        raise ValueError(f"Unknown loss_weighting method: '{self.loss_weighting['method']}'")

    def _forward_step(self, batch, training=True, scaler=None):
        """Common forward step for training and evaluation.

        batch is a dict produced by scale_and_spectrogram_collate_fn with keys:
            'rir'         — [B, T] waveform (collate scaling applied)
            'room_dim'    — [B, 3]
            'mic_loc'     — [B, 3]
            'speaker_loc' — [B, 3]
            'rt60'        — [B]         (when use_rt60_condition=True)
            'dist'        — [B]         (when src_trgt_dist_cond=True; mic–speaker Euclidean distance)
            'images'      — [B, N, 3, H, W]  (SoundSpaces only, optional)
        """
        # ---- prepare spectrogram ----
        rir = batch['rir'].to(self.device)                                # [B, T]
        rir = waveform_to_spectrogram(rir, hop_length=self.hop_length, n_fft=self.n_fft)  # [B, 2, F, T]

        # ---- build scalar conditioning ----
        condition = build_condition_tensor(batch, self.device, self.src_trgt_dist_cond)

        # ---- optional images ----
        images = batch['images'].to(self.device) if 'images' in batch else None

        noise     = torch.randn_like(rir).to(self.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps,
                                  (rir.shape[0],), device=self.device).long()

        noisy_rirs = self.noise_scheduler.add_noise(rir, noise, timesteps)

        if training:
            self.optimizer.zero_grad()

        # ---- forward + loss ----
        if self.use_amp:
            with autocast("cuda"):
                noise_pred = self.model(noisy_rirs, timesteps, condition, images=images)["sample"]
                loss = self._compute_loss(noise_pred, noise, batch, noisy_rirs, timesteps)
            if training:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
        else:
            noise_pred = self.model(noisy_rirs, timesteps, condition, images=images)["sample"]
            loss = self._compute_loss(noise_pred, noise, batch, noisy_rirs, timesteps)
            if training:
                loss.backward()
                self.optimizer.step()

        loss_value      = self.accelerator.gather_for_metrics(loss.detach()).mean().item()
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
        plots_dir = os.path.join(self.logdir, 'plots')
        save_metric(losses_per_epoch['train_loss'], 'log-train-loss', plots_dir, apply_log=True)
        save_metric(losses_per_epoch['train_loss'], 'train-loss', plots_dir, apply_log=False)
        save_metric(losses_per_epoch['train_norm_loss'], 'log-train-norm-loss', plots_dir, apply_log=True)
        save_metric(losses_per_epoch['train_norm_loss'], 'train-norm-loss', plots_dir, apply_log=False)

        save_metric(losses_per_epoch['eval_loss'], 'log-eval-loss', plots_dir, apply_log=True)
        save_metric(losses_per_epoch['eval_loss'], 'eval-loss', plots_dir, apply_log=False)
        save_metric(losses_per_epoch['eval_norm_loss'], 'log-eval-norm-loss', plots_dir, apply_log=True)
        save_metric(losses_per_epoch['eval_norm_loss'], 'eval-norm-loss', plots_dir, apply_log=False)

    def _should_stop_early(self, epoch, best_model_dict):
        """
        Stop training once the eval loss has gone `patience` consecutive epochs without improving.

        All processes reduce the same flag so they leave the epoch loop together.
        """
        stop = False
        if self.accelerator.is_main_process and self.patience and best_model_dict is not None:
            epochs_since_best = epoch + 1 - best_model_dict['epoch']
            stop = epochs_since_best >= self.patience
            if stop:
                msg = (f"Stopping early after {epoch + 1} epochs: {epochs_since_best} without a new "
                       f"best eval loss (patience {self.patience}), best at epoch {best_model_dict['epoch']}.")
                logging.info(msg)
                print(msg)

        flag = torch.tensor([int(stop)], device=self.accelerator.device)
        return bool(self.accelerator.reduce(flag, reduction='max').item())

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