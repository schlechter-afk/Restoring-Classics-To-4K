"""ResNet feature extraction with intermediate layer outputs."""

import os
import gc
from math import ceil
from typing import Tuple, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from torchvision import models
from datasets import load_dataset
from kornia.filters import joint_bilateral_blur
from kornia.color import yuv_to_rgb, rgb_to_yuv

from dataloader_imagenet import ImageNetDataset
from dataloader_resnet import NoisyImageNetDataset

class ResNetFeatureExtractor(nn.Module):
    """Feature extractor using a pre-trained ResNet with intermediate layers."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.init_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # Output: 64x56x56
        self.layer1 = resnet.layer1  # Output: 64x56x56
        self.layer2 = resnet.layer2  # Output: 128x28x28
        self.layer3 = resnet.layer3  # Output: 256x14x14
        self.layer4 = resnet.layer4  # Output: 512x7x7

    def forward(self, x: torch.Tensor) -> dict:
        """Extract features using ResNet.

        Args:
            x: The input tensor of shape (BATCH_SIZE, 3, 224, 224).

        Returns:
            A dictionary containing intermediate feature maps.
        """
        mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std_tensor  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean_tensor) / std_tensor

        res_init = self.init_layer(x)
        res1 = self.layer1(res_init)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)

        return {
            'res1': res1,  # Output: 64x56x56
            'res2': res2,  # Output: 128x28x28
            'res3': res3,  # Output: 256x14x14
            'res4': res4   # Output: 512x7x7
        }


class Colorizer(nn.Module):
    """Colorizer using ResNet features with normalization, dropout, and skip connections."""
    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]] = 5,
            sigma_color: float = 0.1,
            sigma_space: Union[float, Tuple[float, float]] = 1.5,
        ) -> None:
        super().__init__()

        self.feature_extractor = ResNetFeatureExtractor()

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.GroupNorm(16, 32),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.final_layer = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)  # 112x112 -> 224x224

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        self.sigma_color = sigma_color

        if isinstance(sigma_space, float):
            sigma_space = (sigma_space, sigma_space)
        self.sigma_space = sigma_space


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x: The input grayscale tensor of shape (BATCH_SIZE, 1, 224, 224).

        Returns:
            The output tensor of shape (BATCH_SIZE, 2, 224, 224).
        """
        x_original = x.clone()
        x_expanded = x.repeat(1, 3, 1, 1)  # Shape: (BATCH_SIZE, 3, 224, 224)

        features = self.feature_extractor(x_expanded)  # Outputs: 64x112x112, 128x56x56, 256x28x28, 512x7x7
        res1, res2, res3, res4 = features['res1'], features['res2'], features['res3'], features['res4']

        # Decoder with skip connections
        x = self.decoder1(res4)                  # Input: 512x7x7 -> Output: 256x14x14
        x = torch.cat((x, res3), dim=1)          # Skip connection from res3

        x = self.decoder2(x)                     # Input: (256+256)x14x14 -> Output: 128x28x28
        x = torch.cat((x, res2), dim=1)          # Skip connection from res2

        x = self.decoder3(x)                     # Input: (128+128)x28x28 -> Output: 64x56x56
        x = torch.cat((x, res1), dim=1)          # Skip connection from res1

        x = self.decoder4(x)                     # Input: (64+64)x56x56 -> Output: 32x112x112

        x = self.final_layer(x)                  # Input: 32x112x112 -> Output: 2x224x224 (UV channels)

        # x = joint_bilateral_blur(
        #     input=x,
        #     guidance=x_original.repeat(1, 2, 1, 1),  # Repeat grayscale for guidance
        #     kernel_size=self.kernel_size,
        #     sigma_color=self.sigma_color,
        #     sigma_space=self.sigma_space
        # )

        # Scale UV channels to proper range
        u_max = 0.436
        v_max = 0.615
        scale_tensor = torch.tensor([u_max, v_max], device=x.device).view(1, 2, 1, 1)
        x = torch.tanh(x) * scale_tensor

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
    DEVICE             = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    WANDB_RUN_NAME     = f"noiseless_resnet_colorizer_without_JBF_{LR}_{WEIGHT_DECAY}_{VAL_FREQUENCY}"
    CHECKPOINT_DIR     = "/scratch/public_scratch/gp/DIP/checkpoints/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = Colorizer().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    # Make sure to login to HuggingFace before running the code using `huggingface-cli login`
    # You will have to paste the API key from https://huggingface.co/settings/tokens
    # This is needed to access the gated ImageNet dataset

    # train_dataset = load_dataset(
    #     'imagenet-1k', split='train', streaming=True,
    #     cache_dir=CACHE_DIR, trust_remote_code=True
    # ).shuffle()
    # noisy_train_dataset = NoisyImageNetDataset(train_dataset)
    train_dataset = ImageNetDataset('train', cache_dir=CACHE_DIR)

    # val_dataset = load_dataset(
    #     'imagenet-1k', split='validation', streaming=True,
    #     cache_dir=CACHE_DIR, trust_remote_code=True
    # )
    # noisy_val_dataset = NoisyImageNetDataset(val_dataset)
    # val_dataloader = DataLoader(noisy_val_dataset, batch_size=BATCH_SIZE, num_workers=1)
    val_dataset = ImageNetDataset('validation', cache_dir=CACHE_DIR)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=1)

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

        # train_dataset.set_epoch(epoch)
        # noisy_train_dataset.dataset = train_dataset
        # train_dataloader = DataLoader(noisy_train_dataset, batch_size=BATCH_SIZE, num_workers=5)

        train_dataset.set_epoch(epoch)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=5)

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

            colorizer_loss.backward()
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
