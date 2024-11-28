""" Dataloader for the ImageNet dataset."""

from datasets import load_dataset
from torchvision import transforms as T
from torch.utils.data import IterableDataset

class ImageNetDataset(IterableDataset):
    def __init__(self, split: str, cache_dir: str = None):
        super().__init__()

        self.dataset = load_dataset(
            'imagenet-1k', split=split, streaming=True,
            cache_dir=cache_dir, trust_remote_code=True
        )

        if split == 'train':
            self.shuffle()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])


    def shuffle(self) -> None:
        """Shuffle the dataset."""
        self.dataset = self.dataset.shuffle()


    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the dataset.

        Args:
            epoch: The epoch number to set for reshuffling the dataset.
        """
        self.dataset.set_epoch(epoch)


    def __iter__(self):
        for item in self.dataset:
            image = item['image']

            if image.mode == 'L':
                continue

            if image.mode != 'RGB':
                image = image.convert('RGB')

            gray_image = image.convert('L')

            # This transforms from PIL image to torch tensor
            # Values are in the range [0, 1] after the transformation
            image = self.transform(image)            # [3, H, W]
            gray_image = self.transform(gray_image)  # [1, H, W]

            yield {
                'original_rgb': image,
                'denoised_gray': gray_image,
            }


if __name__ == '__main__':
    dataset = ImageNetDataset('train', cache_dir="/scratch/public_scratch/gp/DIP/ImageNet-1k/")
    for sample in dataset:
        rgb = sample['rgb']
        gray = sample['gray']

        print(rgb.shape, gray.shape)
        break
