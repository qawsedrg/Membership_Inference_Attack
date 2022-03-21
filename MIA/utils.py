import multiprocessing
from multiprocessing.pool import ThreadPool
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from snsynth.mwem import MWEMSynthesizer
from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertPreTrainedModel


class trainset(Dataset):
    """
    A simple Dataset wrapper that supports transform
    """

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


def train(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: optimizer,
          criterion: nn.Module, epoches: int, eval: Optional[bool] = False, testloader: Optional[DataLoader] = None) -> \
        Tuple[nn.Module, float, float]:
    r"""
    A typical pytroch training wrapper

    .. note::
        if eval==True, must pass the testloader

    :return: model, training accuracy, evaluation accuracy
    """
    model.to(device)
    model.train()
    acc = None
    val_acc = None
    for epoch in range(epoches):
        epoch_loss = 0
        acc = 0
        with tqdm(enumerate(loader, 0), total=len(loader)) as t:
            for i, data in t:
                correct_items = 0
                data = [d.to(device) for d in data]
                labels = data[-1]

                optimizer.zero_grad()
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = model(*data[:-1])
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits
                if len(outputs.shape) == 2:
                    correct_items += torch.sum(torch.argmax(outputs, dim=-1) == labels).item()
                # ???
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
        acc /= i + 1
    if eval:
        model.eval()
        val_acc = 0
        with torch.no_grad():
            with tqdm(enumerate(testloader, 0), total=len(testloader)) as t:
                for i, data in t:
                    correct_items = 0
                    data = [d.to(device) for d in data]
                    label = data[-1]

                    outputs = model(*data[:-1])
                    if not isinstance(outputs, torch.Tensor):
                        outputs = outputs.logits
                    correct_items += torch.sum(torch.argmax(outputs, dim=-1) == label).item()
                    val_acc_batch = correct_items / testloader.batch_size
                    val_acc += val_acc_batch

                    t.set_description("VAL")
                    t.set_postfix(accuracy="{:.3f}".format(val_acc / (i + 1)))
        val_acc /= i + 1

    return model, acc, val_acc


def forward(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    r"""
    A typical pytroch inference wrapper
    """
    result = torch.Tensor().to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if type(data) is torch.Tensor:
                inputs = data.to(device)
                outputs = model(inputs)
            else:
                # multiple input of model
                inputs = [d.to(device) for d in data[:-1]]
                outputs = model(*inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.logits
            result = torch.cat((result, outputs), dim=0)
    return result


def forward_sklearn(model: BaseEstimator, loader: DataLoader, device: torch.device) -> torch.Tensor:
    r"""
    A typical sklearn inference wrapper

    .. note::
        The input loader and output are torch object
    """
    result = torch.Tensor()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader, 0), total=len(loader)):
            if i > 10:
                break
            if type(data) is torch.Tensor:
                inputs = data.cpu()
                outputs = torch.from_numpy(model.predict_proba(inputs))
            else:
                inputs = [d.cpu() for d in data[:-1]]
                outputs = torch.from_numpy(model.predict_proba(*inputs))
            result = torch.cat((result, outputs), dim=0)
    return result.float().to(device)


class DataStruct():
    def __init__(self, data_in: np.ndarray, data_out: np.ndarray, target_in: np.ndarray, target_out: np.ndarray):
        r"""
        DataStruct to communicate between shadowmodel and attack model

        :param data_in: data trained
        :param data_out: data not trained
        :param target_in: class of data trained
        :param target_out: class of data not trained
        """
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


def get_threshold(membership: np.ndarray, vec: np.ndarray, thresholds: Optional[float] = None) -> Tuple[
    float, Optional[float], float, Optional[float]]:
    r"""
    Given a 1D array data vec,and the membership groud truth (0 or 1) associated, determinate of two thresholds that
    maximize the precision or accuracy

    Or evalute the precision and accuracy if a threshold is provided

    .. note::
        The membership of a data with its vec value greater than threshold is considered 1

    :return: accuracy, threshold that maximizes the accuray, precision, threshold that maximizes the precision
    """
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


def memguard(scores: np.ndarray) -> np.ndarray:
    # Label-Only Membership Inference Attacks (membership-inference-master)
    r"""
    Given confidence vectors, perform memguard post processing to protect from membership inference.

    .. note::
      This defense assumes the strongest defender that can make arbitrary changes to the confidence vector
      so long as it does not change the label. We as well have the (weaker) constrained optimization that will be
      released at a future data.

    :param scores: confidence vectors as 2d numpy array

    :return: 2d scores protected by memguard.

    """
    n_classes = scores.shape[1]
    epsilon = 1e-3
    on_score = (1. / n_classes) + epsilon
    off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
    predicted_labels = np.argmax(scores, axis=-1)
    defended_scores = np.ones_like(scores) * off_score
    defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
    return defended_scores


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, float]:
    r"""
    MixUp

    :param x: data to mix
    :param y: class of data to mix
    :param alpha: parameter of the beta distribution
    :return: X mixed of 2 data, class of the first data, class of the second data, mix ratio
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    r"""
    Modified loss function for MixUp
    """
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mix(X, Y, ratio) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Generate noised data using `SmartNoise <https://github.com/opendp/smartnoise-sdk/tree/main/synth>`_

    :param X: original data
    :param Y: class accosicated
    :param ratio: generated data is ratio times bigger than X
    :return: X generated, Y generated
    """

    data = np.append(X, Y[:, np.newaxis], axis=1)
    df = pd.DataFrame(data)

    synth = MWEMSynthesizer(3.0, 400, 40, 20, split_factor=X.shape[1] + 1, max_bin_count=400)
    synth.fit(df)

    synthetic = synth.sample(int(X.shape[0]) * ratio)
    result = pd.concat([df, synthetic])
    out = np.array(result)

    return out[:, :-1], out[:, -1]


class augmentation_wrapper():
    def __init__(self, aug, n=1):
        self.aug = aug
        self.n = n

    def __call__(self, text):
        # multiprocessing.cpu_count()
        return self.aug.augment(text, n=self.n, num_thread=1)

    def to(self, *args, **kwargs):
        pass