import os.path
import pickle
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from MIA import ShadowModels
from MIA.utils import trainset, get_threshold


class Boundary():
    def __init__(self, shadowmodel: ShadowModels, device: torch.device, transform: Optional = None):
        self.shadowmodel = shadowmodel
        self.device = device
        self.acc_thresh = 0
        self.pre_thresh = 0
        self.transform = transform

    def train(self, show=False) -> Tuple[float, float]:
        if not os.path.exists("./dist_shadow_in") or not os.path.exists("./dist_shadow_out"):
            # todo many shadow models
            dist_shadow_in = self.train_base(self.shadowmodel[0], self.shadowmodel.loader_train)
            dist_shadow_out = self.train_base(self.shadowmodel[0], self.shadowmodel.loader_test)
            pickle.dump(dist_shadow_in, open("./dist_shadow_in", "wb"))
            pickle.dump(dist_shadow_out, open("./dist_shadow_out", "wb"))
        else:
            dist_shadow_in = pickle.load(open("./dist_shadow_in", "rb"))
            dist_shadow_out = pickle.load(open("./dist_shadow_out", "rb"))
        dist_shadow = np.concatenate((dist_shadow_in, dist_shadow_out))
        membership_shadow = np.concatenate((np.ones_like(dist_shadow_in), np.zeros_like(dist_shadow_out)))
        acc, self.acc_thresh, prec, self.pre_thresh = get_threshold(membership_shadow, dist_shadow)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))
        if show:
            right = max(dist_shadow)
            plt.hist(dist_shadow_in, bins=100, range=[0, right], label="in")
            plt.hist(dist_shadow_out, bins=100, range=[0, right], label="out")
            plt.legend()
            plt.show()
        return (acc, prec)

    def train_base(self, model: nn.Module, loader: DataLoader) -> List[float]:
        dist_adv = []
        model.to(self.device)
        with tqdm(enumerate(loader, 0), total=len(loader)) as t:
            for i, data in t:
                xbatch, ybatch = data[0].to(self.device), data[1].to(self.device)
                with torch.no_grad():
                    y_pred = F.softmax(model(xbatch), dim=-1)
                x_selected = xbatch[torch.argmax(y_pred, dim=-1) == ybatch, :]
                dist_adv.extend([0] * (xbatch.shape[0] - x_selected.shape[0]))
                # num_iteration
                x_adv_curr = carlini_wagner_l2(model, x_selected, n_classes=3)
                d = torch.sqrt(torch.sum(torch.square(x_adv_curr - x_selected), dim=1)).cpu().numpy()
                dist_adv.extend(d)
        return dist_adv

    def __call__(self, model, X: torch.Tensor) -> np.ndarray:
        x_adv_curr = carlini_wagner_l2(model, X, n_classes=3)
        d = torch.sqrt(torch.sum(torch.square(x_adv_curr - X), dim=(1, 2, 3))).cpu().numpy()
        return d > self.acc_thresh

    def evaluate(self, target: Optional[nn.Module] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None) -> Tuple[float, float]:
        if not os.path.exists("./dist_target_in") or not os.path.exists("./dist_target_out"):
            loader_in = DataLoader(trainset(X_in, Y_in, self.transform), batch_size=64, shuffle=False)
            loader_out = DataLoader(trainset(X_out, Y_out, self.transform), batch_size=64, shuffle=False)
            dist_target_in = self.train_base(loader_in, target)
            dist_target_out = self.train_base(loader_out, target)
            pickle.dump(dist_target_in, open("./dist_target_in", "wb"))
            pickle.dump(dist_target_out, open("./dist_target_out", "wb"))
        else:
            dist_target_in = pickle.load(open("./dist_target_in", "rb"))
            dist_target_out = pickle.load(open("./dist_target_out", "rb"))
        dist_target = np.concatenate((dist_target_in, dist_target_out))
        membership_target = np.concatenate((np.ones_like(dist_target_in), np.zeros_like(dist_target_out)))
        acc, _, _, _ = get_threshold(membership_target, dist_target, self.acc_thresh)
        _, _, prec, _ = get_threshold(membership_target, dist_target, self.pre_thresh)
        print("test_acc:{:},test_pre:{:}".format(acc, prec))
        return (acc, prec)
