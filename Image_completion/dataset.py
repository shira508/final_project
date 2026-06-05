import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class InpaintingDataset(Dataset):

    def __init__(self, corrupted_dir, mask_dir, original_dir):

        self.corrupted_dir = corrupted_dir
        self.mask_dir = mask_dir
        self.original_dir = original_dir

        self.filenames = [
            f for f in sorted(os.listdir(corrupted_dir))
            if os.path.exists(os.path.join(mask_dir, f))
            and os.path.exists(os.path.join(original_dir, f))
        ]

        self.transform_image = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        f = self.filenames[idx]

        corrupted = Image.open(
            os.path.join(self.corrupted_dir, f)
        ).convert("L")

        mask = Image.open(
            os.path.join(self.mask_dir, f)
        ).convert("L")

        original = Image.open(
            os.path.join(self.original_dir, f)
        ).convert("L")

        corrupted = self.transform_image(corrupted)
        original = self.transform_image(original)

        mask = self.transform_mask(mask)

        # בינארי
        mask = (mask > 0.5).float()

        return corrupted, mask, original