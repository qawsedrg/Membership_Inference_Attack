import os.path
import pickle
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from MIA import ShadowModels
from MIA.utils import trainset, get_threshold


class Noise():
    def __init__(self, shadowmodel: ShadowModels, device: torch.device, transform: Optional = None):
        self.shadowmodel = shadowmodel
        self.device = device
        self.acc_thresh = 0
        self.pre_thresh = 0
        self.stddev = np.linspace(2, 10, 100)
        self.noisesamples = 100
        self.transform = transform

    def train(self, show=False) -> Tuple[float, float]:
        if not os.path.exists("./dist_shadow_in_noise") or not os.path.exists("./dist_shadow_out_noise"):
            # todo many shadow models
            dist_shadow_in = self.train_base(self.shadowmodel.loader_train, self.shadowmodel[0], self.stddev,
                                             self.noisesamples)
            dist_shadow_out = self.train_base(self.shadowmodel.loader_test, self.shadowmodel[0], self.stddev,
                                              self.noisesamples)
            pickle.dump(dist_shadow_in, open("./dist_shadow_in_noise", "wb"))
            pickle.dump(dist_shadow_out, open("./dist_shadow_out_noise", "wb"))
        else:
            dist_shadow_in = pickle.load(open("./dist_shadow_in_noise", "rb"))
            dist_shadow_out = pickle.load(open("./dist_shadow_out_noise", "rb"))
        dist_shadow = np.concatenate((dist_shadow_in, dist_shadow_out))
        membership_shadow = np.concatenate((np.ones_like(dist_shadow_in), np.zeros_like(dist_shadow_out)))
        if show:
            right = max(dist_shadow)
            plt.hist(dist_shadow_in, bins=100, range=[0, right], label="in")
            plt.hist(dist_shadow_out, bins=100, range=[0, right], label="out")
            plt.legend()
            plt.show()
        acc, self.acc_thresh, prec, self.pre_thresh = get_threshold(membership_shadow, dist_shadow)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))
        return (acc, prec)

    def train_base(self, loader, model, stddev, noise_samples) -> List[int]:
        num_in = []
        if isinstance(model, nn.Module):
            model.to(self.device)
        with tqdm(enumerate(loader, 0), total=len(loader)) as t:
            for _, data in t:
                xbatch, ybatch = data[0].to(self.device), data[1].to(self.device)
                with torch.no_grad():
                    y_pred = F.softmax(model(xbatch), dim=-1) if isinstance(model, nn.Module) else torch.from_numpy(
                        model.predict_proba(xbatch.cpu())).to(self.device)
                x_selected = xbatch[torch.argmax(y_pred, dim=-1) == ybatch, :]
                y_selected = ybatch[torch.argmax(y_pred, dim=-1) == ybatch]
                num_in.extend([0] * (xbatch.shape[0] - x_selected.shape[0]))
                # num_iteration
                for i in range(x_selected.shape[0]):
                    n = 0
                    for dev in stddev:
                        noise = torch.from_numpy(dev * np.random.randn(noise_samples, *x_selected.shape[1:])).to(
                            self.device)
                        # 注意范围
                        # x_noisy = torch.clamp(x_selected[i, :] + noise, 0, 1).float()
                        x_noisy = (x_selected[i, :] + noise).float()
                        b_size = 100
                        with torch.no_grad():
                            for j in range(noise_samples // b_size + 1):
                                y_pred = F.softmax(model(x_noisy[j * b_size:(j + 1) * b_size]), dim=-1) if isinstance(
                                    model, nn.Module) else torch.from_numpy(
                                    model.predict_proba(x_noisy[j * b_size:(j + 1) * b_size])).to(self.device)
                                n += torch.sum(torch.argmax(y_pred, dim=-1) == y_selected[i]).item()
                    num_in.append(n / noise_samples)
        return num_in

    def evaluate(self, target: Optional = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None) -> Tuple[float, float]:
        if not os.path.exists("./dist_target_in_noise") or not os.path.exists("./dist_target_out_noise"):
            loader_in = DataLoader(trainset(X_in, Y_in, self.transform), batch_size=64, shuffle=False)
            loader_out = DataLoader(trainset(X_out, Y_out, self.transform), batch_size=64, shuffle=False)
            dist_target_in = self.train_base(loader_in, target, self.stddev,
                                             self.noisesamples)
            dist_target_out = self.train_base(loader_out, target, self.stddev,
                                              self.noisesamples)
            pickle.dump(dist_target_in, open("./dist_target_in_noise", "wb"))
            pickle.dump(dist_target_out, open("./dist_target_out_noise", "wb"))
        else:
            dist_target_in = pickle.load(open("./dist_target_in_noise", "rb"))
            dist_target_out = pickle.load(open("./dist_target_out_noise", "rb"))
        dist_target = np.concatenate((dist_target_in, dist_target_out))
        membership_target = np.concatenate((np.ones_like(dist_target_in), np.zeros_like(dist_target_out)))
        acc, _, _, _ = get_threshold(membership_target, dist_target, self.acc_thresh)
        _, _, prec, _ = get_threshold(membership_target, dist_target, self.pre_thresh)
        print("test_acc:{:},test_pre:{:}".format(acc, prec))
        return (acc, prec)

    def __call__(self, model, X: torch.Tensor) -> np.ndarray:
        y = torch.argmax(F.softmax(model(X), dim=-1), dim=-1)
        num_in = []
        for i in range(X.shape[0]):
            n = 0
            for dev in self.stddev:
                noise = torch.from_numpy(dev * np.random.randn(self.noisesamples, *X.shape[1:])).to(self.device)
                ##
                x_noisy = torch.clamp(X[i, :] + noise, 0, 1).float()
                b_size = 100
                with torch.no_grad():
                    for j in range(self.noisesamples // b_size + 1):
                        y_pred = F.softmax(model(x_noisy[j * b_size:(j + 1) * b_size]), dim=-1)
                        n += torch.sum(torch.argmax(y_pred, dim=-1) == y[i]).item()
            num_in.append(n / self.noisesamples)
        return np.array(num_in) > self.acc_thresh
