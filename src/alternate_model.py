"""Modified implementation of the colorizer from "Deep Colorization", by Cheng et al. (2015)."""

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
from kornia.feature import SIFTDescriptor
from kornia.filters import joint_bilateral_blur
from kornia.color import yuv_to_rgb, rgb_to_yuv

from dataloader_resnet import NoisyImageNetDataset

class Colorizer(nn.Module):
    """Model that colorizes the grayscale images."""
    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]] = 5,
            sigma_color: float = 0.1,
            sigma_space: Union[float, Tuple[float, float]] = 1.5,
        ) -> None:
        super().__init__()

        self.sift_7x7 = SIFTDescriptor(patch_size=32)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.sift_14x14 = SIFTDescriptor(patch_size=16)
        self.sift_skip_conv1 = nn.Conv2d(128, 32, kernel_size=1)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.sift_28x28 = SIFTDescriptor(patch_size=8)
        self.sift_skip_conv2 = nn.Conv2d(128, 16, kernel_size=1)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.GroupNorm(num_groups=2, num_channels=8),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.GroupNorm(num_groups=1, num_channels=4),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.final_layer = nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)  # 112x112 -> 224x224

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        self.sigma_color = sigma_color

        if isinstance(sigma_space, float):
            sigma_space = (sigma_space, sigma_space)
        self.sigma_space = sigma_space


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: The input tensor of shape (batch_size, 1, 224, 224).

        Returns:
            The output tensor of shape (batch_size, 2, 224, 224).
        """
        # original_x represents the luminance values of the image (Y channel)
        original_x = x.clone()

        image_7x7 = original_x.clone()
        patches_7x7 = F.unfold(image_7x7, kernel_size=7, stride=7)  # (batch_size, 49, 32*32)
        patches_7x7 = patches_7x7.view(-1, 1, 32, 32)  # (batch_size*49, 1, 32, 32)

        sift_feats_7x7 = self.sift_7x7(patches_7x7)  # (batch_size*49, 128)
        sift_feats_7x7 = sift_feats_7x7.view(-1, 128, 7, 7)  # (batch_size, 128, 7, 7)

        x = self.decoder1(sift_feats_7x7)  # (batch_size, 32, 14, 14)

        image_14x14 = original_x.clone()
        patches_14x14 = F.unfold(image_14x14, kernel_size=14, stride=14)  # (batch_size, 196, 16*16)
        patches_14x14 = patches_14x14.view(-1, 1, 16, 16)  # (batch_size*196, 1, 16, 16)

        sift_feats_14x14 = self.sift_14x14(patches_14x14)  # (batch_size*196, 128)
        sift_feats_14x14 = sift_feats_14x14.view(-1, 128, 14, 14)  # (batch_size, 128, 14, 14)

        x = self.decoder2(torch.cat((x, self.sift_skip_conv1(sift_feats_14x14)), dim=1))  # (batch_size, 16, 28, 28)

        image_28x28 = original_x.clone()
        patches_28x28 = F.unfold(image_28x28, kernel_size=28, stride=28)  # (batch_size, 784, 8*8)
        patches_28x28 = patches_28x28.view(-1, 1, 8, 8)  # (batch_size*784, 1, 8, 8)

        sift_feats_28x28 = self.sift_28x28(patches_28x28)  # (batch_size*784, 128)
        sift_feats_28x28 = sift_feats_28x28.view(-1, 128, 28, 28)  # (batch_size, 128, 28, 28)

        x = self.decoder3(torch.cat((x, self.sift_skip_conv2(sift_feats_28x28)), dim=1))  # (batch_size, 8, 56, 56)

        x = self.decoder4(x)  # (batch_size, 4, 112, 112)

        x = self.final_layer(x)  # (batch_size, 2, 224, 224)

        x = joint_bilateral_blur(
            input=x,
            guidance=original_x.repeat(1, 2, 1, 1),
            kernel_size=self.kernel_size,
            sigma_color=self.sigma_color,
            sigma_space=self.sigma_space
        )  # (batch_size, 2, 270, 512)

        u_max = 0.436
        v_max = 0.615
        scale_tensor = torch.tensor([u_max, v_max], device=x.device)

        x = torch.tanh(x) * scale_tensor.view(1, 2, 1, 1)  # (batch_size, 2, 270, 512)

        # x represents the U and V channels of the image
        return x

    @staticmethod
    def uv_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Converts the YUV channels to RGB channels.

        Args:
            y: The Y channel of shape (BATCH_SIZE, 1, 224, 224).
            uv: The UV channels of shape (BATCH_SIZE, 2, 224, 224).

        Returns:
            The RGB image of shape (BATCH_SIZE, 3, 224, 224).
        """
        yuv_image = torch.cat((y, uv), dim=1)
        return yuv_to_rgb(yuv_image)


if __name__ == "__main__":
    LR                 = 1e-4
    DEVICE             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WANDB_LOG          = True
    CACHE_DIR          = "/scratch/public_scratch/gp/DIP/ImageNet-1k/"
    NUM_EPOCHS         = 5
    BATCH_SIZE         = 128
    WEIGHT_DECAY       = 1e-2
    VAL_FREQUENCY      = 1000
    SCHEDULER_FACTOR   = 0.8
    SCHEDULER_PATIENCE = 2
    COLORIZER_WEIGHT   = 0.75
    UPSCALER_WEIGHT    = 1 - COLORIZER_WEIGHT
    WANDB_RUN_NAME     = f"sift_colorizer_with_JBF_{LR}_{WEIGHT_DECAY}_{VAL_FREQUENCY}"
    CHECKPOINT_DIR     = "/scratch/public_scratch/gp/DIP/checkpoints/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = Colorizer().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    # Make sure to login to HuggingFace before running the code using `huggingface-cli login`
    # You will have to paste the API key from https://huggingface.co/settings/tokens
    # This is needed to access the gated ImageNet dataset

    train_dataset = load_dataset(
        'imagenet-1k', split='train', streaming=True,
        cache_dir=CACHE_DIR, trust_remote_code=True
    ).shuffle()
    noisy_train_dataset = NoisyImageNetDataset(train_dataset)

    val_dataset = load_dataset(
        'imagenet-1k', split='validation', streaming=True,
        cache_dir=CACHE_DIR, trust_remote_code=True
    )
    noisy_val_dataset = NoisyImageNetDataset(val_dataset)
    val_dataloader = DataLoader(noisy_val_dataset, batch_size=BATCH_SIZE, num_workers=1)

    best_val_loss = float('inf')

    if WANDB_LOG:
        wandb.init(project="RestoringClassicsGlory", name=WANDB_RUN_NAME)
        wandb.watch(model, log="all")

    # From here: https://huggingface.co/datasets/ILSVRC/imagenet-1k#data-splits
    TRAIN_DATASET_SIZE = 1281167
    NUM_TRAIN_BATCHES  = ceil(TRAIN_DATASET_SIZE / BATCH_SIZE)
    for epoch in range(NUM_EPOCHS):
        model.train()
        batch_count = 0
        epoch_colorizer_loss = 0.0

        train_dataset.set_epoch(epoch)
        noisy_train_dataset.dataset = train_dataset
        train_dataloader = DataLoader(noisy_train_dataset, batch_size=BATCH_SIZE, num_workers=5)

        pbar = tqdm(train_dataloader, total=NUM_TRAIN_BATCHES)
        for batch in pbar:
            pbar.set_description_str(f"Epoch {epoch+1}")
            pbar.refresh()
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            denoised_gray = batch['denoised_gray'].to(DEVICE)  # Shape: (BATCH_SIZE, 1, 224, 224)
            original_rgb  = batch['original_rgb'].to(DEVICE)   # Shape: (BATCH_SIZE, 3, 224, 224)

            uv_channels = model(denoised_gray)

            original_yuv = rgb_to_yuv(original_rgb)
            colorizer_loss = nn.MSELoss()(uv_channels, original_yuv[:, 1:])
            epoch_colorizer_loss += colorizer_loss.item()

            colorizer_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_count += 1

            if WANDB_LOG:
                wandb.log({
                    "train_loss": colorizer_loss.item(),
                    "avg_train_loss": epoch_colorizer_loss / batch_count,
                })

            pbar.set_postfix_str(f"Colorizer Loss: {epoch_colorizer_loss / batch_count:.4f}")

            if batch_count % VAL_FREQUENCY == 0:
                model.eval()
                val_batch_count = 0
                avg_colorizer_loss = 0.0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        pbar.set_postfix_str(f"Colorizer Loss: {epoch_colorizer_loss / batch_count:.4f}, Val Batch: {val_batch_count + 1}")

                        val_denoised_gray = val_batch['denoised_gray'].to(DEVICE)
                        val_original_rgb  = val_batch['original_rgb'].to(DEVICE)

                        val_uv = model(val_denoised_gray)

                        val_yuv = rgb_to_yuv(val_original_rgb)
                        val_colorizer_loss = nn.MSELoss()(val_uv, val_yuv[:, 1:])

                        avg_colorizer_loss += val_colorizer_loss.item()

                        val_batch_count += 1

                avg_colorizer_loss = avg_colorizer_loss / val_batch_count if val_batch_count > 0 else float('inf')

                if WANDB_LOG:
                    wandb.log({"val_loss": avg_colorizer_loss})

                print(f"Validation loss after {batch_count} batches: {val_colorizer_loss:.4f}\n")

                scheduler.step(avg_colorizer_loss)

                if avg_colorizer_loss < best_val_loss:
                    best_val_loss = avg_colorizer_loss

                    checkpoint_path = os.path.join(
                        CHECKPOINT_DIR,
                        f"{WANDB_RUN_NAME}_epoch_{epoch+1}_batch_{batch_count}.pth"
                    )

                    torch.save({
                        'epoch': epoch+1,
                        'batch': batch_count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_colorizer_loss,
                    }, checkpoint_path)

                model.train()

    if WANDB_LOG:
        wandb.finish()
