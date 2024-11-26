"""Data loader for Noisy ImageNet dataset."""

import os
import random
from typing import Dict

import cv2
import torch
import numpy as np
from PIL import Image

from skimage.transform import resize
from skimage.util import random_noise

from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset

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
        """Add noise to the input image.

        Args:
            image: Input image as a numpy array.
            noise_type: Type of noise to add to the image.

        Returns:
            Noisy image as a numpy array.
        """
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
        """Denoise the input image.

        Args:
            image: Noisy image as a numpy array.
            noise_type: Type of noise added to the image.

        Returns:
            Denoised image as a numpy array.
        """
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
            image_np = cv2.resize(image_np, (224, 224))  # Resize to (224, 224)

            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)

            original_rgb = image_np.astype(np.float32) / 255.0
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            image_gray = image_gray.astype(np.float32) / 255.0

            noise_type = random.choice(list(self.noise_params.keys()))
            noisy_image = self.add_noise(image_gray, noise_type)
            denoised_image = self.denoise_image(noisy_image, noise_type)

            original_rgb_tensor = torch.FloatTensor(original_rgb).permute(2, 0, 1)

            denoised_image_tensor = torch.FloatTensor(denoised_image).unsqueeze(0)

            yield {
                'original_rgb': original_rgb_tensor,
                'denoised_gray': denoised_image_tensor,
            }


class NewNoisyImageDataset:
    """Helper class to prepare the Noisy ImageNet dataset."""
    def __init__(self):
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
        """Add noise to the input image.

        Args:
            image: Input image as a numpy array.
            noise_type: Type of noise to add to the image.

        Returns:
            Noisy image as a numpy array.
        """
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
        """Denoise the input image.

        Args:
            image: Noisy image as a numpy array.
            noise_type: Type of noise added to the image.

        Returns:
            Denoised image as a numpy array.
        """
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


    def process_sample(self, sample) -> Dict[str, torch.Tensor]:
        """Add noise and denoise the image from the dataset.

        Args:
            sample: An object containing the image.

        Returns:
            A dictionary containing the original RGB image and the denoised grayscale image.
        """
        image: Image = sample['image']

        if image.mode == 'L':
            # Skip grayscale images
            return None

        if image.mode == 'RGBA':
            # Ignore alpha channel
            image = image.convert('RGB')

        image = image.resize((224, 224))

        gray_image = image.convert('L')

        original_image = np.array(image).astype(np.float32) / 255.0
        gray_image     = np.array(gray_image).astype(np.float32) / 255.0

        noise_type     = random.choice(list(self.noise_params.keys()))
        noisy_image    = self.add_noise(gray_image, noise_type)
        denoised_image = self.denoise_image(noisy_image, noise_type)

        original_image = torch.FloatTensor(original_image).permute(2, 0, 1)
        denoised_image = torch.FloatTensor(denoised_image).unsqueeze(0)

        return {
            'original_rgb': original_image,
            'denoised_gray': denoised_image,
        }


    def prepare_dataset(self, name: str, split: str, cache_dir: str):
        """Prepare the Noisy ImageNet dataset.

        Args:
            name: Name of the dataset.
            split: Split of the dataset.
            cache_dir: Directory to cache the dataset.

        Returns:
            An iterable dataset containing the original RGB images and the denoised grayscale images.
        """
        dataset = load_dataset(name, split=split, streaming=True, trust_remote_code=True, cache_dir=cache_dir)
        dataset = dataset.remove_columns('label')
        dataset = dataset.map(self.process_sample)
        dataset = dataset.filter(lambda x: x is not None)
        return dataset
