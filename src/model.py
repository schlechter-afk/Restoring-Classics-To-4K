"""Code for the entire model that colorizes and upscales the images."""

import gc
from typing import Tuple, Union

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

from colorizer import Colorizer
from upscaler import Upscaler

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
    LR     = 1e-3
    # torch.autograd.set_detect_anomaly(True)

    model = Restorer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pbar = tqdm(range(1000))
    for _ in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        batch = torch.rand(16, 1, 270, 512).to(DEVICE)
        y = model(batch)

        coloured_batch = batch.repeat(1, 3, 1, 1)

        # Bilinear interpolation
        gt = F.interpolate(coloured_batch, size=(2160, 4096), mode="bilinear", align_corners=False)

        # Bicubic interpolation
        # Mentioned as an issue in the documentation that this method can overshoot values
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        # gt = F.interpolate(coloured_batch, size=(2160, 4096), mode="bicubic", align_corners=False)
        # gt = gt.clamp(0, 1)

        loss = nn.MSELoss()(y, gt)
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        loss.backward()
