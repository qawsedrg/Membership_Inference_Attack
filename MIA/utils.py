import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


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
        epoch_loss = 0
        acc = 0
        with tqdm(enumerate(loader, 0), total=len(loader)) as t:
            for i, data in t:
                correct_items = 0
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if len(outputs.shape) == 2:
                    correct_items += torch.sum(torch.argmax(outputs, axis=-1) == labels).item()
                elif len(outputs.shape) == 1:
                    correct_items += torch.sum(outputs[labels == 1] > .5).item()
                    correct_items += torch.sum(outputs[labels == 0] < .5).item()
                else:
                    raise
                acc_batch = correct_items / loader.batch_size
                acc += acc_batch
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                t.set_description("Epoch {:}/{:} Train".format(epoch + 1, epoches))
                t.set_postfix(accuracy="{:.3f}".format(acc / (i + 1)), loss="{:.3f}".format(epoch_loss / (i + 1)))
    return model


def forward(model, loader, device):
    result = torch.Tensor().to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if type(data) is torch.Tensor:
                inputs = data.to(device)
            else:
                inputs = data[0].to(device)
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
        self.fc1 = nn.Linear(n, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.squeeze(nn.Sigmoid()(self.fc2(x)), dim=-1)
        return x


def get_threshold(membership, vec, thresholds=None):
    if thresholds is None:
        fpr, tpr, thresholds = roc_curve(membership, vec)
    accuracy_scores = []
    precision_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(membership, (vec > thresh).astype(int)))
        precision_scores.append(precision_score(membership, (vec > thresh).astype(int)))
    accuracies = np.array(accuracy_scores)
    precisions = np.array(precision_scores)
    max_accuracy = accuracies.max()
    max_precision = precisions.max()
    max_accuracy_threshold = thresholds[accuracies.argmax()]
    max_precision_threshold = thresholds[precisions.argmax()]
    return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold
