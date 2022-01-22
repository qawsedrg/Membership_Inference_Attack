import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import multiprocessing
import nlpaug.augmenter.word as naw


class trainset(Dataset):
    def __init__(self, X, Y=None, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        if self.Y is None:
            return self.transform(self.X[index]) if self.transform is not None else self.X[index]
        else:
            return self.transform(self.X[index]) if self.transform is not None else self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


def train(model, loader, device, optimizer, criterion, epoches, verbose=True):
    model.train()
    model.to(device)
    for epoch in range(epoches):
        epoch_loss = 0
        acc = 0
        if verbose:
            with tqdm(enumerate(loader, 0), total=len(loader)) as t:
                for i, data in t:
                    correct_items = 0
                    data = [d.to(device) for d in data]
                    labels = data[-1]

                    optimizer.zero_grad()
                    outputs = model(*data[:-1])
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
        else:
            for i, data in enumerate(loader, 0):
                data = [d.to(device) for d in data]
                labels = data[-1]
                optimizer.zero_grad()
                outputs = model(*data[:-1])
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    return model


def forward(model, loader, device):
    result = torch.Tensor().to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if type(data) is torch.Tensor:
                inputs = data.to(device)
                outputs = model(inputs)
            else:
                inputs = [d.to(device) for d in data[:-1]]
                outputs = model(*inputs)
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
    accuracy_scores = []
    precision_scores = []
    if thresholds is None:
        def f(thresholds):
            for thresh in thresholds:
                accuracy_scores.append(accuracy_score(membership, (vec > thresh).astype(int)))
                precision_scores.append(precision_score(membership, (vec > thresh).astype(int)))
                accuracies = np.array(accuracy_scores)
                precisions = np.array(precision_scores)
            return accuracies, precisions

        fpr, tpr, thresholds = roc_curve(membership, vec)
        numberOfThreads = multiprocessing.cpu_count()
        pool = ThreadPool(processes=numberOfThreads)
        Chunks = np.array_split(thresholds, numberOfThreads)
        results = pool.map_async(f, Chunks)
        pool.close()
        pool.join()
        for result in results.get():
            accuracies, precisions = result
            accuracy_scores.extend(accuracies)
            precision_scores.extend(precisions)
        accuracies = np.array(accuracy_scores)
        precisions = np.array(precision_scores)
        max_accuracy = accuracies.max()
        max_precision = precisions.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        max_precision_threshold = thresholds[precisions.argmax()]
        return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold
    else:
        accuracy_scores.append(accuracy_score(membership, (vec > thresholds).astype(int)))
        precision_scores.append(precision_score(membership, (vec > thresholds).astype(int)))
        accuracies = np.array(accuracy_scores)
        precisions = np.array(precision_scores)
        max_accuracy = accuracies.max()
        max_precision = precisions.max()
        return max_accuracy, None, max_precision, None


def memguard(scores):
    # Label-Only Membership Inference Attacks (membership-inference-master)
    """ Given confidence vectors, perform memguard post processing to protect from membership inference.

    Note that this defense assumes the strongest defender that can make arbitrary changes to the confidence vector
    so long as it does not change the label. We as well have the (weaker) constrained optimization that will be
    released at a future data.

    Args:
      scores: confidence vectors as 2d numpy array

    Returns: 2d scores protected by memguard.

    """
    n_classes = scores.shape[1]
    epsilon = 1e-3
    on_score = (1. / n_classes) + epsilon
    off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
    predicted_labels = np.argmax(scores, axis=-1)
    defended_scores = np.ones_like(scores) * off_score
    defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
    return defended_scores


class augmentation_wrapper():
    def __init__(self, aug, n):
        self.aug = aug
        self.n = n

    def __call__(self, text):
        return self.aug(text, n=self.n, num_thread=multiprocessing.cpu_count())