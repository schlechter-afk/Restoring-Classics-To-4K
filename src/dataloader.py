import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import cv2
from typing import Tuple, Dict
import random
from skimage.util import random_noise
from skimage.transform import resize
from datasets import load_dataset
import os
import sys

os.environ['HF_DATASETS_CACHE'] = '/scratch/swayam/.cache/huggingface/datasets/'

class NoisyImageNetDataset(IterableDataset):
    def __init__(self, dataset, image_size=(512, 270)):
        self.dataset = dataset
        self.image_size = image_size
        
        # Noise parameters for adding noise to images
        self.noise_params = {
            'salt_and_pepper': {'amount': 0.07},
            'gaussian': {'var': 0.01},
            'fine_grained': {'val': 0.036}
        }

        # Noise filter mapping for denoising images: (filter_type, filter_params)
        self.noise_filter_mapping = {
            'salt_and_pepper': ('fastNLM', {'h': 40}),
            'gaussian': ('fastNLM', {'h': 15}),
            'fine_grained': ('fastNLM', {'h': 17})
        }


    def add_noise(self, image: np.ndarray, noise_type: str) -> np.ndarray:
        noisy_image = image.copy()
        params = self.noise_params[noise_type]
        
        if noise_type == 'salt_and_pepper':
            noisy_image = random_noise(image, mode='s&p', amount=params['amount'])
            
        elif noise_type == 'gaussian':
            noisy_image = random_noise(image, mode='gaussian', var=params['var'])
            
        elif noise_type == 'fine_grained':
            val = 0.036
            rows, cols = image.shape

            # Full resolution
            noise_im1 = np.zeros((rows, cols))
            noise_im1 = random_noise(noise_im1, mode='gaussian', var=val**2, clip=False)

            # Half resolution
            noise_im2 = np.zeros((rows//2, cols//2))
            noise_im2 = random_noise(noise_im2, mode='gaussian', var=(val*2)**2, clip=False)  # Use val*2 (needs tuning...)
            noise_im2 = resize(noise_im2, (rows, cols))  # Upscale to original image size

            noise_im = noise_im1 + noise_im2 
            noisy_img = noisy_image + noise_im

            noisy_image = np.clip(noisy_img, 0, 1)
            
        return noisy_image
    

    def denoise_image(self, image: np.ndarray, noise_type: str) -> np.ndarray:
        filter_type, filter_params = self.noise_filter_mapping[noise_type]

        # Convert image to uint8
        image_uint8 = (image * 255).astype(np.uint8)

        if filter_type == 'median':
            denoised_uint8 = cv2.medianBlur(image_uint8, filter_params['ksize'])
        elif filter_type == 'gaussian':
            denoised_uint8 = cv2.GaussianBlur(image_uint8, filter_params['ksize'], filter_params['sigmaX'])
        elif filter_type == 'fastNLM':
            denoised_uint8 = cv2.fastNlMeansDenoising(image_uint8, h=filter_params['h'])
        else:
            denoised_uint8 = image_uint8

        # Convert back to float in [0, 1]
        denoised_image = denoised_uint8.astype(np.float32) / 255.0
        return denoised_image


    def __iter__(self):
        for item in self.dataset:
            image = item['image']

            if image.mode == 'L': 
                # Skip grayscale images
                continue

            if image.mode == 'RGBA':
                # Ignore alpha channel
                image = image.convert('RGB')

            image_np = np.array(image)  # Shape: (H, W, 3)
            image_np = cv2.resize(image_np, self.image_size)  # Resize to (512, 270)

            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)

            original_rgb = image_np.transpose(2, 0, 1)  # Shape: (3, H, W)
            # image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            # image_gray is obtained by converting RGB to YUV and taking the Y channel
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)[:, :, 0]

            original_rgb = original_rgb.astype(np.float32) / 255.0
            image_gray = image_gray.astype(np.float32) / 255.0

            noise_type = random.choice(list(self.noise_params.keys()))
            noisy_image = self.add_noise(image_gray, noise_type)
            denoised_image = self.denoise_image(noisy_image, noise_type)

            original_rgb_tensor = torch.FloatTensor(original_rgb)
            denoised_image_tensor = torch.FloatTensor(denoised_image).unsqueeze(0)
            image_gray_tensor = torch.FloatTensor(image_gray).unsqueeze(0)

            yield {
                'original_rgb': original_rgb_tensor,
                'denoised_gray': image_gray_tensor,
            }