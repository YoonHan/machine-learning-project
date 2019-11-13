import os
import torch
import pandas as pd
from PIL import Image
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
        painting = Image.open(painting_path).convert('RGB')

        sample = {'painting': painting, 'label': self.painting_frame.iloc[idx, 1]}

        if self.transform:
            sample['painting'] = self.transform(sample['painting'])

        return sample

    def __len__(self):
        return len(self.painting_frame)


def show_batch(batch_idx, sample_batched):
    """Show image for batch of samples."""
    print(f'batch #{batch_idx}')
    paintings_batch, label_batch = sample_batched['painting'], sample_batched['label']
    print(paintings_batch.size())



painting_dataset = PaintingDataset(csv_file_path='./painting_label.csv',
                                   root_dir='../data/training/',
                                   transform=transforms.Compose([
                                       transforms.Resize((256, 256)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                   ]))

painting_dataloader = DataLoader(painting_dataset, batch_size=32)

for batch_idx, sample_batched in enumerate(painting_dataloader):
    show_batch(batch_idx, sample_batched)

    if batch_idx == 8:
        break

