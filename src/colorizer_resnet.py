"""ResNet feature extraction with intermediate layer outputs."""
import gc
from typing import Tuple, Union

import torch
from torch import nn, optim
from tqdm import tqdm
from kornia.color import yuv_to_rgb, rgb_to_yuv
from kornia.feature import DenseSIFTDescriptor
from kornia.filters import joint_bilateral_blur

import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    """Feature extractor using a pre-trained ResNet with intermediate layers."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

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
        kernel_size: int = 5,
        sigma_color: float = 0.1,
        sigma_space: float = 1.5,
    ) -> None:
        super().__init__()

        self.feature_extractor = ResNetFeatureExtractor()

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.GroupNorm(16, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.final_layer = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)  # 112x112 -> 224x224

        self.kernel_size = (kernel_size, kernel_size)
        self.sigma_color = sigma_color
        self.sigma_space = (sigma_space, sigma_space)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        uv = joint_bilateral_blur(
            input=x,
            guidance=x_original.repeat(1, 2, 1, 1),  # Repeat grayscale for guidance
            kernel_size=self.kernel_size,
            sigma_color=self.sigma_color,
            sigma_space=self.sigma_space
        )

        # Scale UV channels to proper range
        u_max = 0.436
        v_max = 0.615
        scale_tensor = torch.tensor([u_max, v_max], device=uv.device).view(1, 2, 1, 1)
        uv = torch.tanh(uv) * scale_tensor

        return uv


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR     = 1e-3

    model = Colorizer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pbar = tqdm(range(100))
    for _ in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        batch   = torch.rand(16, 1, 224, 224).to(DEVICE)
        rand_gt = torch.rand(16, 2, 224, 224).to(DEVICE)

        y = model(batch)

        loss = nn.MSELoss()(y, rand_gt)
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()