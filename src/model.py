import gc
import os
import copy
from typing import Tuple, Union

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb  
from dotenv import load_dotenv

from colorizer import Colorizer
from upscaler import Upscaler
from dataloader import NoisyImageNetDataset  
from datasets import load_dataset

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
        self.upscaler = Upscaler()

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: The input tensor of shape (batch_size, 1, 270, 512).

        Returns:
            The output tensor of shape (batch_size, 3, 2160, 4096).
        """
        x = self.colorizer(x)  # (batch_size, 3, 270, 512)
        x = self.upscaler(x)   # (batch_size, 3, 2160, 4096)

        return x

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR = 1e-3
    num_epochs = 1  
    batch_size = 4
    validation_computation_steps = 10000
    val_batches = 100

    os.environ['HF_DATASETS_CACHE'] = '/scratch/swayam/.cache/huggingface/datasets/'

    load_dotenv()

    model = Restorer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    HF_TOKEN = os.getenv("HF_TOKEN")

    train_dataset = load_dataset('imagenet-1k', split='train', streaming=True, token=HF_TOKEN)
    val_dataset = load_dataset('imagenet-1k', split='validation', streaming=True, token=HF_TOKEN)

    our_train_dataset = NoisyImageNetDataset(train_dataset)
    our_val_dataset = NoisyImageNetDataset(val_dataset)

    train_dataloader = DataLoader(our_train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(our_val_dataset, batch_size=batch_size)

    best_val_loss = float('inf')
    batch_count = 0

    wandb.init(project="RestoringClassicsGlory", name="train_v1")
    wandb.watch(model, log="all")

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            denoised_gray = batch['denoised_gray'].to(DEVICE)   # Shape: (batch_size, 1, 270, 512)
            original_rgb = batch['original_rgb'].to(DEVICE)     # Shape: (batch_size, 3, 270, 512)

            y = model(denoised_gray)                            # Shape: (batch_size, 3, 2160, 4096)

            gt = F.interpolate(original_rgb,                    # Shape: (batch_size, 3, 2160, 4096)
                                size=(2160, 4096), 
                                mode="bilinear", 
                                align_corners=False)

            loss = nn.MSELoss()(y, gt)
            loss.backward()
            optimizer.step()

            batch_count += 1

            wandb.log({"train_loss": loss.item(), "batch": batch_count})

            pbar.set_postfix_str(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

            # Every 10k batches, compute validation loss
            if batch_count % validation_computation_steps == 0:
                model.eval()
                val_losses = []
                val_batch_count = 0
                max_val_batches = val_batches  

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_denoised_gray = val_batch['denoised_gray'].to(DEVICE)
                        val_original_rgb = val_batch['original_rgb'].to(DEVICE)

                        val_y = model(val_denoised_gray)
                        val_gt = F.interpolate(val_original_rgb, size=(2160, 4096), mode="bilinear", align_corners=False)

                        val_loss = nn.MSELoss()(val_y, val_gt)
                        val_losses.append(val_loss.item())

                        val_batch_count += 1
                        if val_batch_count >= max_val_batches:
                            break

                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')

                wandb.log({"val_loss": avg_val_loss, "batch": batch_count})

                print(f"Validation Loss after {batch_count} batches: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:

                    best_val_loss = avg_val_loss

                    checkpoint_path = os.path.join("checkpoints", f"model_epoch_{epoch+1}_batch_{batch_count}.pth")

                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

                    torch.save({
                        'epoch': epoch+1,
                        'batch': batch_count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                    }, checkpoint_path)

                    print(f"Model saved to {checkpoint_path}")

                model.train()  # Set back to training mode

    wandb.finish()