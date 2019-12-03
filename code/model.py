import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BATCH_SIZE = 1

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
        label = torch.LongTensor([self.painting_frame.iloc[idx, 1]])

        sample = {'painting': painting, 'label': label}

        if self.transform:
            sample['painting'] = self.transform(sample['painting'])

        return sample

    def __len__(self):
        return len(self.painting_frame)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        print('Constructing Network...')
        self.conv1 = nn.Conv2d(3, 6, 3)  # in_channels, out_channels, kernel_size, stride...
        self.pool = nn.MaxPool2d(2, 2)   # kernel width, height
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(BATCH_SIZE * 16 * 53 * 53, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 30)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, BATCH_SIZE * 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def print_batch(batch_idx, s_batched):
    """Show image for batch of samples."""
    print(f'batch #{batch_idx}')
    paintings_batch, label_batch = s_batched['painting'], s_batched['label']
    print(paintings_batch.size())


painting_dataset = PaintingDataset(csv_file_path='./painting_label.csv',
                                   root_dir='../data/training/',
                                   transform=transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                   ]))

trainloader = DataLoader(painting_dataset, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    net = Net()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
        print('Epoch %d ...' % (epoch + 1))
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # input data
            print(f'batch(image)#{i + 1} processing...')
            input, label = data['painting'], data['label'].view(1)
            # print('input shape:', input.shape)
            # initialize gradient
            optimizer.zero_grad

            # forward -> backward -> gradient update(optimization)
            output = net(input)
            # print('output shape:', output.shape)
            print('label:', label)
            l = loss(output, label)
            l.backward()
            optimizer.step()

            # printing process
            running_loss += l.item()
            if i % 1 == 0:
                print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, l.item()))
                # running_loss = 0.0

    print('Finished training !!!')