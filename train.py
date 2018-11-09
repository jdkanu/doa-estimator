import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from tensorboardX import SummaryWriter
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from itertools import compress
import argparse

# dataset
class CustomDataset(Dataset):
    def __len__(self):
        return 3000

    def __getitem__(self, index):
        data = np.random.uniform(0, 1, [6, 25, 513]).astype('f')
        label = np.atleast_2d(0)
        return data, label

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (bsz, 64, 25, 64)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (bsz, 64, 25, 8)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),  # (bsz, 64, 25, 2)
        )

        # from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
        self.hidden_size = 64
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        out = self.conv(x)  # (bsz, 64, 25, 2)
        reshape = out.permute(0, 2, 1, 3).contiguous().view(32, 25, 128)
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        
        return out

if __name__ == "__main__":
    # initialize dataset
    train_dataset = CustomDataset()

    # initialize Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True, num_workers=1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(device)
    for i, (data, labels) in enumerate(train_loader):
        outputs = model(data)
        print(str(outputs))
