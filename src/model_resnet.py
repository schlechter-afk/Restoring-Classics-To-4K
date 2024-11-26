"""Code for the entire model that colorizes and upscales the images."""

import os
import gc
from math import ceil
from typing import Tuple, Union

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from datasets import load_dataset
from kornia.color import yuv_to_rgb, rgb_to_yuv

from colorizer_resnet import Colorizer
from dataloader_resnet import NoisyImageNetDataset

from perceptual_loss import PerceptualLoss

class Restorer(nn.Module):
    """Model that colorizes and upscales the images."""
    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]] = 5,
            sigma_color: float = 0.1,
            sigma_space: Union[float, Tuple[float, float]] = 1.5,
        ) -> None:
        super().__init__()

        self.colorizer = Colorizer(kernel_size, sigma_color, sigma_space)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: The input tensor of shape (BATCH_SIZE, 1, 270, 512).
        Returns:
            - uv_channels: The predicted UV channels of shape (BATCH_SIZE, 2, 270, 512).
        """
        uv_channels = self.colorizer(x)  # Shape: (BATCH_SIZE, 2, 270, 512)
        return uv_channels


if __name__ == "__main__":
    LR                 = 1e-4
    DEVICE             = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    WANDB_LOG          = True
    CACHE_DIR          = "/scratch/swayam/other_stuff/DIP/ImageNet-1k/"
    NUM_EPOCHS         = 3
    BATCH_SIZE         = 64
    WEIGHT_DECAY       = 1e-2
    VAL_FREQUENCY      = 2500
    MAX_VAL_BATCHES    = 100
    SCHEDULER_FACTOR   = 0.8
    SCHEDULER_PATIENCE = 2
    WANDB_RUN_NAME     = f"perceptual_loss_w_resnet_backbone_{LR}_{WEIGHT_DECAY}_{VAL_FREQUENCY}"
    CHECKPOINT_DIR     = "/scratch/swayam/other_stuff/DIP/checkpoints/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = Restorer().to(DEVICE)
    perceptual_loss_fn = PerceptualLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    # Make sure to login to HuggingFace before running the code using `huggingface-cli login`
    # You will have to paste the API key from https://huggingface.co/settings/tokens
    # This is needed to access the gated ImageNet dataset

    train_dataset = load_dataset(
        'imagenet-1k', split='train', streaming=True,
        cache_dir=CACHE_DIR, trust_remote_code=True,
    ).shuffle()
    noisy_train_dataset = NoisyImageNetDataset(train_dataset)
    train_dataloader = DataLoader(noisy_train_dataset, batch_size=BATCH_SIZE, num_workers=5)

    val_dataset = load_dataset(
        'imagenet-1k', split='validation', streaming=True,
        cache_dir=CACHE_DIR, trust_remote_code=True
    )
    noisy_val_dataset = NoisyImageNetDataset(val_dataset)
    val_dataloader = DataLoader(noisy_val_dataset, batch_size=BATCH_SIZE, num_workers=1)

    best_val_loss = float('inf')
    batch_count   = 0

    # Define weighting factors
    MSE_WEIGHT = 1.0
    PERCEPTUAL_WEIGHT = 0.0005  # Adjust this based on experiments

    if WANDB_LOG:
        wandb.init(project="RestoringClassicsGlory", name=WANDB_RUN_NAME)
        wandb.watch(model, log="all")

    # From here: https://huggingface.co/datasets/ILSVRC/imagenet-1k#data-splits
    TRAIN_DATASET_SIZE = 1281167
    NUM_TRAIN_BATCHES  = ceil(TRAIN_DATASET_SIZE / BATCH_SIZE)
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_mse_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", total=NUM_TRAIN_BATCHES)
        for batch in pbar:
            pbar.refresh()
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            denoised_gray = batch['denoised_gray'].to(DEVICE)  # Shape: (BATCH_SIZE, 1, 270, 512)
            original_rgb  = batch['original_rgb'].to(DEVICE)   # Shape: (BATCH_SIZE, 3, 270, 512)

            uv_channels = model(denoised_gray)  # Shape: (BATCH_SIZE, 2, 270, 512)

            original_yuv = rgb_to_yuv(original_rgb)

            pred_yuv = torch.cat([denoised_gray, uv_channels], dim=1)
            pred_rgb = yuv_to_rgb(pred_yuv)
            target_rgb = original_rgb

            mse_loss = nn.MSELoss()(uv_channels, original_yuv[:, 1:])
            perceptual_loss = perceptual_loss_fn(pred_rgb, target_rgb)

            loss = MSE_WEIGHT * mse_loss + PERCEPTUAL_WEIGHT * perceptual_loss

            # loss = nn.MSELoss()(uv_channels, original_yuv[:, 1:])
            epoch_mse_loss += mse_loss.item()
            epoch_perceptual_loss += perceptual_loss.item()
            epoch_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_count += 1

            if WANDB_LOG:
                wandb.log({
                    "train_loss": loss.item(),
                    "avg_train_loss": epoch_train_loss / batch_count,
                    "mse_loss": mse_loss.item(),
                    "avg_mse_loss": epoch_mse_loss / batch_count,
                    "perceptual_loss": perceptual_loss.item(),
                    "avg_perceptual_loss": epoch_perceptual_loss / batch_count,
                })

            pbar.set_postfix_str(f"Avg Train Loss: {epoch_train_loss / batch_count:.4f}, Avg MSE Loss: {epoch_mse_loss / batch_count:.4f}, Avg Perceptual Loss: {epoch_perceptual_loss / batch_count:.4f}")

            # Every 10k batches, compute validation loss
            if batch_count % VAL_FREQUENCY == 0:
                model.eval()
                avg_val_loss    = 0.0
                val_batch_count = 0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_denoised_gray = val_batch['denoised_gray'].to(DEVICE)
                        val_original_rgb  = val_batch['original_rgb'].to(DEVICE)

                        val_uv_channels = model(val_denoised_gray)
                        val_yuv = rgb_to_yuv(val_original_rgb)
                        # val_loss = nn.MSELoss()(val_uv_channels, val_yuv[:, 1:])
                        # avg_val_loss += val_loss.item()

                        val_pred_yuv = torch.cat([val_denoised_gray, val_uv_channels], dim=1)
                        val_pred_rgb = yuv_to_rgb(val_pred_yuv)

                        val_perceptual_loss = perceptual_loss_fn(val_pred_rgb, val_original_rgb)

                        val_mse_loss = nn.MSELoss()(val_uv_channels, val_yuv[:, 1:])
                        val_loss = MSE_WEIGHT * val_mse_loss + PERCEPTUAL_WEIGHT * val_perceptual_loss

                        avg_val_loss += val_loss.item()

                        val_batch_count += 1

                        # if val_batch_count >= MAX_VAL_BATCHES:
                        #     break

                avg_val_loss = avg_val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                if WANDB_LOG:
                    wandb.log({"val_loss": avg_val_loss})

                print(f"Validation Loss after {batch_count} batches: {avg_val_loss:.4f}")

                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

                    checkpoint_path = os.path.join(
                        CHECKPOINT_DIR,
                        f"{WANDB_RUN_NAME}_epoch_{epoch+1}_batch_{batch_count}.pth"
                    )
                    torch.save({
                        'epoch': epoch+1,
                        'batch': batch_count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                    }, checkpoint_path)

                model.train()

    if WANDB_LOG:
        wandb.finish()
