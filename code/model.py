import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PaintingDataset(Dataset):
    def __init__(self, root_dir, csv_file_path, transform = None):
        self.painting_frame = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        painting_path = os.path.join(self.root_dir, self.painting_frame.iloc[idx, 0])
        # painting = io.imread(painting_path)
        painting = Image.open(painting_path).convert('RGB')
        # painting = color.gray2rgb(painting)     # Convert gray scale to rgb

        sample = {'painting': painting, 'label': self.painting_frame.iloc[idx, 1]}

        if self.transform:
            sample['painting'] = self.transform(sample['painting'])

        return sample

    def __len__(self):
        return len(self.painting_frame)


painting_dataset = PaintingDataset(csv_file_path='./painting_label.csv',
                                   root_dir='../data/training/',
                                   transform=transforms.Compose([
                                       transforms.Resize((256, 256)),
                                       transforms.ToTensor(),
                                   ]))

for i in range(len(painting_dataset)):
    sample = painting_dataset[i]
    print(sample)

    if i == 3:
        break