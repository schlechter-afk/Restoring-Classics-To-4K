"""Code for the upscaling model that upscales the colorized images to 4K resolution."""

import os
import gc
from math import ceil

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from datasets import load_dataset

from dataloader_imagenet import ImageNetDataset
from dataloader_resnet import NoisyImageNetDataset

class Upscaler(nn.Module):
    """Model that upscales the colorized images to 4K resolution."""
    def __init__(self):
        super().__init__()

        self.transposed_conv1 = nn.ConvTranspose2d(3,  8,  kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=8)
        self.transposed_conv2 = nn.ConvTranspose2d(8,  16, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=16)
        self.transposed_conv3 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.transposed_conv4 = nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.transposed_conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.transposed_conv6 = nn.ConvTranspose2d(32, 8,  kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.GroupNorm(num_groups=4, num_channels=8)
        self.transposed_conv7 = nn.ConvTranspose2d(8,  3,  kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

        # 1x1 Convolutions for Residual Connections
        self.skip_conv1 = nn.Conv2d(3, 8, kernel_size=1)    # For first residual connection
        self.skip_conv2 = nn.Conv2d(8, 32, kernel_size=1)   # For second residual connection
        self.skip_conv3 = nn.Conv2d(32, 64, kernel_size=1)  # For third residual connection
        self.skip_conv4 = nn.Conv2d(64, 8, kernel_size=1)   # For fourth residual connection

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass of the model.

        Args:
            x: The input tensor of shape (batch_size, 3, 270, 512).

        Returns:
            The output tensor of shape (batch_size, 3, 2160, 4096).
        """
        original_x = x.clone()

        x = self.relu(self.norm1(self.transposed_conv1(x)))
        x = self.dropout(x)
        # skip_conv1_out = self.skip_conv1(
        #                     F.interpolate(
        #                         original_x,
        #                         scale_factor=2,
        #                         mode="bilinear",
        #                         align_corners=False
        #                     )
        #                 )
        skip_conv1_out = F.interpolate(
                            self.skip_conv1(original_x),
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=False
                        )
        x += skip_conv1_out
        first_2x = x.clone()

        x = self.relu(self.norm2(self.transposed_conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.norm3(self.transposed_conv3(x)))
        x = self.dropout(x)
        skip_conv2_out = self.skip_conv2(first_2x)
        x += skip_conv2_out
        blk1_out = x.clone()

        x = self.relu(self.norm4(self.transposed_conv4(x)))
        x = self.dropout(x)
        # skip_conv3_out = self.skip_conv3(
        #                     F.interpolate(
        #                         blk1_out,
        #                         scale_factor=2,
        #                         mode="bilinear",
        #                         align_corners=False
        #                     )
        #                 )
        skip_conv3_out = F.interpolate(
                            self.skip_conv3(blk1_out),
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=False
                        )
        x += skip_conv3_out
        second_2x = x.clone()

        x = self.relu(self.norm5(self.transposed_conv5(x)))
        x = self.dropout(x)
        x = self.relu(self.norm6(self.transposed_conv6(x)))
        x = self.dropout(x)
        skip_conv4_out = self.skip_conv4(second_2x)
        x += skip_conv4_out
        x = self.transposed_conv7(x)

        # Normalize the output so that the final image has values in [0, 1] range
        # x = (x - x.min()) / (x.max() - x.min() + 1e-6) # removing normalizing since groupnorm is used
        x = x.clamp(0, 1)

        return x

    # def forward(self, x):
    #     """Forward pass of the model.

    #     Args:
    #         x: The input tensor of shape (batch_size, 3, 270, 512).

    #     Returns:
    #         The output tensor of shape (batch_size, 3, 2160, 4096).
    #     """
    #     x = self.relu(self.transposed_conv1(x))  # (batch_size, 8,  540,  1024)
    #     x = self.relu(self.transposed_conv2(x))  # (batch_size, 16, 540,  1024)
    #     x = self.relu(self.transposed_conv3(x))  # (batch_size, 32, 540,  1024)
    #     x = self.relu(self.transposed_conv4(x))  # (batch_size, 64, 1080, 2048)
    #     x = self.relu(self.transposed_conv5(x))  # (batch_size, 32, 1080, 2048)
    #     x = self.relu(self.transposed_conv6(x))  # (batch_size, 8,  1080, 2048)
    #     x = self.transposed_conv7(x)             # (batch_size, 3,  2160, 4096)

    #     # Normalize the output so that the final image has values in [0, 1] range
    #     x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    #     x = x.clamp(0, 1)

    #     return x

if __name__ == "__main__":
    LR                 = 1e-4
    DEVICE             = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    WANDB_LOG          = False
    CACHE_DIR          = "/scratch/public_scratch/gp/DIP/ImageNet-1k/"
    NUM_EPOCHS         = 5
    BATCH_SIZE         = 8
    WEIGHT_DECAY       = 1e-2
    VAL_FREQUENCY      = 10000
    SCHEDULER_FACTOR   = 0.8
    SCHEDULER_PATIENCE = 2
    COLORIZER_WEIGHT   = 0.75
    UPSCALER_WEIGHT    = 1 - COLORIZER_WEIGHT
    WANDB_RUN_NAME     = f"upscaler_{LR}_{WEIGHT_DECAY}_{VAL_FREQUENCY}"
    CHECKPOINT_DIR     = "/scratch/public_scratch/gp/DIP/checkpoints/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = Upscaler().to(DEVICE)
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
        epoch_upscaler_loss = 0.0

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

            original_rgb = batch['original_rgb']  # Shape: (BATCH_SIZE, 3, 224, 224)
            interpolated_rgb = F.interpolate(original_rgb, size=(270, 512), mode="bilinear", align_corners=False).to(DEVICE)

            upscaled_rgb_pred = model(interpolated_rgb)
            upscaled_rgb_gt   = F.interpolate(original_rgb, size=(2160, 4096), mode="bilinear", align_corners=False).to(DEVICE)

            upscaler_loss = nn.MSELoss()(upscaled_rgb_pred, upscaled_rgb_gt)
            epoch_upscaler_loss += upscaler_loss.item()

            upscaler_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_count += 1

            if WANDB_LOG:
                wandb.log({
                    "train_loss": upscaler_loss.item(),
                    "avg_train_loss": epoch_upscaler_loss / batch_count,
                })

            pbar.set_postfix_str(f"Colorizer Loss: {epoch_upscaler_loss / batch_count:.4f}")

            if batch_count % VAL_FREQUENCY == 0:
                model.eval()
                val_batch_count = 0
                avg_upscaler_loss = 0.0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        pbar.set_postfix_str(f"Colorizer Loss: {epoch_upscaler_loss / batch_count:.4f}, Val Batch: {val_batch_count + 1}")

                        val_original_rgb  = val_batch['original_rgb']
                        val_interpolated_rgb = F.interpolate(val_original_rgb, size=(270, 512), mode="bilinear", align_corners=False).to(DEVICE)

                        val_upscaled_rgb_pred = model(val_interpolated_rgb)
                        val_upscaled_rgb_gt   = F.interpolate(val_original_rgb, size=(2160, 4096), mode="bilinear", align_corners=False).to(DEVICE)

                        val_upscaler_loss = nn.MSELoss()(val_upscaled_rgb_pred, val_upscaled_rgb_gt)
                        avg_upscaler_loss += val_upscaler_loss.item()

                        val_batch_count += 1

                avg_upscaler_loss = avg_upscaler_loss / val_batch_count if val_batch_count > 0 else float('inf')

                if WANDB_LOG:
                    wandb.log({"val_loss": avg_upscaler_loss})

                print(f"Validation loss after {batch_count} batches: {avg_upscaler_loss:.4f}\n")

                scheduler.step(avg_upscaler_loss)

                if avg_upscaler_loss < best_val_loss:
                    best_val_loss = avg_upscaler_loss

                    checkpoint_path = os.path.join(
                        CHECKPOINT_DIR,
                        f"{WANDB_RUN_NAME}_epoch_{epoch+1}_batch_{batch_count}.pth"
                    )

                    torch.save({
                        'epoch': epoch+1,
                        'batch': batch_count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_upscaler_loss,
                    }, checkpoint_path)

                model.train()

    if WANDB_LOG:
        wandb.finish()
