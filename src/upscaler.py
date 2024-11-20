"""Code for the upscaling model that upscales the colorized images to 4K resolution."""

import gc

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

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

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: The input tensor of shape (batch_size, 3, 270, 512).

        Returns:
            The output tensor of shape (batch_size, 3, 2160, 4096).
        """
        x = self.relu(self.norm1(self.transposed_conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.transposed_conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.norm3(self.transposed_conv3(x)))
        x = self.dropout(x)
        x = self.relu(self.norm4(self.transposed_conv4(x)))
        x = self.dropout(x)
        x = self.relu(self.norm5(self.transposed_conv5(x)))
        x = self.dropout(x)
        x = self.relu(self.norm6(self.transposed_conv6(x)))
        x = self.dropout(x)
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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR     = 1e-3

    model = Upscaler().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pbar = tqdm(range(1000))
    for _ in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        batch = torch.rand(16, 3, 270, 512).to(DEVICE)
        y = model(batch)

        # Bilinear interpolation
        gt = F.interpolate(batch, size=(2160, 4096), mode="bilinear", align_corners=False)

        # Bicubic interpolation
        # Mentioned as an issue in the documentation that this method can overshoot values
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        # gt = F.interpolate(batch, size=(2160, 4096), mode="bicubic", align_corners=False)
        # gt = gt.clamp(0, 1)

        loss = nn.MSELoss()(y, gt)
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        loss.backward()

        optimizer.step()
