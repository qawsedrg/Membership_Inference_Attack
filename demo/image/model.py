import torch
import torch.nn.functional as F
from torch import nn


class CIFAR(nn.Module):
    def __init__(self, n, p=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, n)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(torch.tanh(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x
