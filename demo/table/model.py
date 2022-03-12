import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def __init__(self, in_n, out_n=100, p=0):
        super().__init__()
        self.fc1 = nn.Linear(in_n, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, out_n)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(self.fc3(x))
        return x
