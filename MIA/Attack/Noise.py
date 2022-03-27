import os.path
import pickle
from typing import Optional, Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from MIA import ShadowModels
from MIA.utils import trainset, get_threshold


class Noise():
    def __init__(self, shadowmodel: ShadowModels, stddev: np.ndarray, noisesamples: int, device: torch.device,
                 transform: Optional = None):
        r"""
        Noise Attack model

        Test the robutness of victime model when adding gaussien noise to the data

        Compute a historam of number of data - the number of stddev with which the data is correctly classified

        Deduce a simple threshold (of number of stddev) that maximizes the accuracy or precision

        .. note::
            The choice of stddev can affect greatly the result.

            It should be chosen as that victime model classifies only a part of (of stddev) the noised data incorrectly (not all not none)

        .. warning::
            if needed, torch.clamp should be used to make sure that the noised data is within the valid range


        :param shadowmodel: shadowmodel
        :param stddev: the stddev value of the noise to be added to the data
        :param noisesamples: number of noise to test for each stddev, to reduce the volatility of result
        :param device: torch.device object
        :param transform: transformation to perform on images
        """
        self.shadowmodel = shadowmodel
        self.device = device
        self.acc_thresh = 0
        self.pre_thresh = 0
        # np.linspace(2, 10, 100) for iris
        self.stddev = stddev
        self.noisesamples = noisesamples
        self.transform = transform

    def train(self, show: Optional[bool] = False, reuse: Optional[bool] = True) -> Tuple[float, float]:
        # store the calculated number, should be modified if needed
        # in - trained, out - not trained
        if not os.path.exists("./num_shadow_in_noise") or not os.path.exists(
                "./num_shadow_out_noise") or reuse == False:
            num_shadow_in = self.train_base(self.shadowmodel.loader_train, self.shadowmodel[0], self.stddev,
                                            self.noisesamples)
            num_shadow_out = self.train_base(self.shadowmodel.loader_test, self.shadowmodel[0], self.stddev,
                                             self.noisesamples)
            pickle.dump(num_shadow_in, open("./num_shadow_in_noise", "wb"))
            pickle.dump(num_shadow_out, open("./num_shadow_out_noise", "wb"))
        else:
            num_shadow_in = pickle.load(open("./num_shadow_in_noise", "rb"))
            num_shadow_out = pickle.load(open("./num_shadow_out_noise", "rb"))
        num_shadow = np.concatenate((num_shadow_in, num_shadow_out))
        membership_shadow = np.concatenate((np.ones_like(num_shadow_in), np.zeros_like(num_shadow_out)))
        if show:
            right = max(num_shadow)
            plt.hist(num_shadow_in, bins=100, range=[0, right], label="in")
            plt.hist(num_shadow_out, bins=100, range=[0, right], label="out")
            plt.legend()
            plt.show()
        acc, self.acc_thresh, prec, self.pre_thresh = get_threshold(membership_shadow, num_shadow)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))
        return (acc, prec)

    def train_base(self, loader: DataLoader, model: Union[BaseEstimator, nn.Module], stddev: np.ndarray,
                   noise_samples: int) -> List[int]:
        r"""

        :return:  List of number of stddev with which the noise data is classfied correctly
        """
        num_in = []
        if isinstance(model, nn.Module):
            model.to(self.device)
        with tqdm(enumerate(loader, 0), total=len(loader)) as t:
            for _, data in t:
                xbatch, ybatch = data[0].to(self.device), data[1].to(self.device)
                with torch.no_grad():
                    # should be changed if softmax is performed in the model
                    y_pred = F.softmax(model(xbatch), dim=-1) if isinstance(model, nn.Module) else torch.from_numpy(
                        model.predict_proba(xbatch.cpu())).to(self.device)
                # misclassififed exemple set directly to 0 (so that they will be classified as no trained)
                x_selected = xbatch[torch.argmax(y_pred, dim=-1) == ybatch, :]
                y_selected = ybatch[torch.argmax(y_pred, dim=-1) == ybatch]
                num_in.extend([0] * (xbatch.shape[0] - x_selected.shape[0]))
                for i in range(x_selected.shape[0]):
                    n = 0
                    for dev in stddev:
                        noise = torch.from_numpy(dev * np.random.randn(noise_samples, *x_selected.shape[1:])).to(
                            self.device)
                        # range! attention!
                        # x_noisy = torch.clamp(x_selected[i, :] + noise, 0, 1).float()
                        x_noisy = (x_selected[i, :] + noise).float()
                        b_size = 100
                        # processes the noisy data by batch of size b_size if noisesamples is big
                        with torch.no_grad():
                            for j in range(noise_samples // b_size + 1):
                                # should be changed if softmax is performed in the model
                                y_pred = F.softmax(model(x_noisy[j * b_size:(j + 1) * b_size]), dim=-1) if isinstance(
                                    model, nn.Module) else torch.from_numpy(
                                    model.predict_proba(x_noisy[j * b_size:(j + 1) * b_size])).to(self.device)
                                n += torch.sum(torch.argmax(y_pred, dim=-1) == y_selected[i]).item()
                    num_in.append(n / noise_samples)
        return num_in

    def evaluate(self, target: Optional = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None, reuse: Optional[bool] = True) -> Tuple[float, float]:
        if not os.path.exists("./num_target_in_noise") or not os.path.exists(
                "./num_target_out_noise") or reuse == False:
            loader_in = DataLoader(trainset(X_in, Y_in, self.transform), batch_size=64, shuffle=False)
            loader_out = DataLoader(trainset(X_out, Y_out, self.transform), batch_size=64, shuffle=False)
            num_target_in = self.train_base(loader_in, target, self.stddev,
                                            self.noisesamples)
            num_target_out = self.train_base(loader_out, target, self.stddev,
                                             self.noisesamples)
            pickle.dump(num_target_in, open("./num_target_in_noise", "wb"))
            pickle.dump(num_target_out, open("./num_target_out_noise", "wb"))
        else:
            num_target_in = pickle.load(open("./num_target_in_noise", "rb"))
            num_target_out = pickle.load(open("./num_target_out_noise", "rb"))
        num_target = np.concatenate((num_target_in, num_target_out))
        membership_target = np.concatenate((np.ones_like(num_target_in), np.zeros_like(num_target_out)))
        acc, _, _, _ = get_threshold(membership_target, num_target, self.acc_thresh)
        _, _, prec, _ = get_threshold(membership_target, num_target, self.pre_thresh)
        print("test_acc:{:},test_pre:{:}".format(acc, prec))
        return (acc, prec)

    def __call__(self, model, X: torch.Tensor) -> np.ndarray:
        y = torch.argmax(F.softmax(model(X), dim=-1), dim=-1)
        num_in = []
        for i in range(X.shape[0]):
            n = 0
            for dev in self.stddev:
                noise = torch.from_numpy(dev * np.random.randn(self.noisesamples, *X.shape[1:])).to(self.device)
                # range! attention!
                # x_noisy = torch.clamp(X[i, :] + noise, 0, 1).float()
                x_noisy = (X[i, :] + noise).float()
                b_size = 100
                with torch.no_grad():
                    for j in range(self.noisesamples // b_size + 1):
                        # should be changed if softmax is performed in the model
                        y_pred = F.softmax(model(x_noisy[j * b_size:(j + 1) * b_size]), dim=-1)
                        n += torch.sum(torch.argmax(y_pred, dim=-1) == y[i]).item()
            num_in.append(n / self.noisesamples)
        # can be set to pre_thresh if needed
        return np.array(num_in) > self.acc_thresh
