import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class trainset(Dataset):
    def __init__(self, X, Y=None, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        if self.Y is None:
            return self.transform(self.X[index, :]) if self.transform is not None else self.X[index, :]
        else:
            return self.transform(self.X[index, :]) if self.transform is not None else self.X[index, :], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


def train(model, loader, device, optimizer, criterion, epoches):
    model.train()
    model.to(device)
    for epoch in range(epoches):

        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    return model


def forward(model, loader, device):
    result = torch.Tensor().to(device)
    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        result = torch.cat((result, outputs), dim=0)
    return result


class DataStruct():
    def __init__(self, data_in, data_out, target_in, target_out):
        self.data_in = data_in
        self.data_out = data_out
        self.target_in = target_in
        self.target_out = target_out


class attackmodel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc1 = nn.Linear(n, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
