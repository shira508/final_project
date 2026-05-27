from torchvision import datasets
from torchvision.transforms import v2
import torch
import matplotlib.pyplot as plt

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((128,128)),
    v2.ToDtype(torch.float32, scale=True)
])

training_data = datasets.ImageFolder(
    root=r"C:\Projects\FinalProject\NewProject\dataset1",
    transform=transform
)

image, label = training_data[0]

print(image.shape)
print(label)

plt.imshow(image.permute(1,2,0))
plt.show()


import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label