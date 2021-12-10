import os.path
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader

from MIA.ShadowModels import ShadowModels
from MIA.utils import trainset, train, attackmodel, forward, get_threshold


class ConfidenceVector():
    def __init__(self, shadowmodel: ShadowModels, epoches: int, device: torch.device, topx: Optional[int] = -1):
        self.shadowdata = shadowmodel.data
        self.n_classes = int(max(torch.max(self.shadowdata.target_in).cpu().numpy(),
                                 torch.max(self.shadowdata.target_out).cpu().numpy()) + 1)
        self.topx = topx
        self.epoches = epoches
        self.device = device

    def train(self, show=False):
        self.attack_models = []
        if self.topx == -1:
            for i in range(self.n_classes):
                train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                     self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
                train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                     torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(
                    self.device)
                attack_model = attackmodel(train_x.shape[-1])
                attack_model.to(self.device)
                optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
                loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
                print("\nTraining attack model for class {:}".format(i))
                attack_model = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                     epoches=self.epoches)
                self.attack_models.append(attack_model)
                fig = plt.figure()
                if show:
                    train_x_in = self.shadowdata.data_in[self.shadowdata.target_in == i].cpu()
                    train_x_out = self.shadowdata.data_in[self.shadowdata.target_in == i].cpu()
                    X_in_tsne = TSNE(n_components=2).fit_transform(train_x_in)
                    X_out_tsne = TSNE(n_components=2).fit_transform(train_x_out)

                    ax = fig.add_subplot()

                    ax.scatter(X_out_tsne[:, 0], X_out_tsne[:, 1], marker='^', label="Not Trained")
                    ax.scatter(X_in_tsne[:, 0], X_in_tsne[:, 1], marker='o', label="Trained")

                    ax.set_xlabel('X_{:} Label'.format(i))
                    ax.set_ylabel('Y_{:} Label'.format(i))
                    ax.legend()

                    plt.show()
        else:
            train_x = torch.sort(torch.cat((self.shadowdata.data_in, self.shadowdata.data_out), dim=0), dim=-1)[0][:,
                      -self.topx:].to(self.device)
            train_y = torch.cat(
                (torch.ones(self.shadowdata.data_in.shape[0]), torch.zeros(self.shadowdata.data_out.shape[0]))).to(
                self.device)
            attack_model = attackmodel(train_x.shape[-1])
            attack_model.to(self.device)
            optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
            loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
            attack_model = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                 epoches=self.epoches)
            self.attack_models.append(attack_model)
            fig = plt.figure()

    def __call__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        if self.topx == -1:
            classes = torch.max(X, dim=-1).indices
            result = torch.Tensor().to(self.device)
            data_x = torch.Tensor().to(self.device)
            data_y = torch.Tensor().to(self.device)
            for i in range(self.n_classes):
                x = X[classes == i]
                with torch.no_grad():
                    result = torch.cat((result, self.attack_models[i](x)))
                data_x = torch.cat((data_x, x), dim=0)
                if Y is not None:
                    y = Y[classes == i]
                    data_y = torch.cat((data_y, y), dim=0)
            return data_x, data_y, result
        else:
            with torch.no_grad():
                return X, Y, self.attack_models[0](X)

    def show(self):
        # todo
        data_in = torch.sort(self.shadowdata.data_in, dim=-1)[0][:, -self.topx:].cpu()
        data_out = torch.sort(self.shadowdata.data_out, dim=-1)[0][:, -self.topx:].cpu()
        attack_model = self.attack_models[0]

        ax = plt.axes(projection='3d')

        xp = np.linspace(0, 1, 100)
        yp = np.linspace(0, 1, 100)
        zp = np.linspace(0, 1, 100)
        x1, y1, z1 = np.meshgrid(xp, yp, zp)
        xyz = np.c_[x1.ravel(), y1.ravel(), z1.ravel()]
        with torch.no_grad():
            y_pred = attack_model(torch.from_numpy(xyz).to(self.device).float()).cpu().detach().numpy().reshape(
                x1.shape)
        x1, y1 = np.meshgrid(xp, yp)
        z = np.argmax((y_pred > 0.5).astype(int), axis=-1) / 100
        ax.contourf(x1, y1, z)

        ax.scatter3D(data_out[:, 0], data_out[:, 1], data_out[:, 2], marker='^', label="Not Trained")
        ax.scatter3D(data_in[:, 0], data_in[:, 1], data_in[:, 2], marker='o', label="Trained")

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()

        plt.show()

    def evaluate(self, target: Optional[nn.Module] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None):

        if target is not None:
            if self.topx == -1:
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                target.to(self.device)
                loader = DataLoader(trainset(X_in, None, transform), batch_size=64, shuffle=False)
                output_in = forward(target, loader, self.device)
                loader = DataLoader(trainset(X_out, None, transform), batch_size=64, shuffle=False)
                output_out = forward(target, loader, self.device)
                _, _, result_in = self(F.softmax(output_in, dim=-1))
                _, _, result_out = self(F.softmax(output_out, dim=-1))
                correct = 0
                correct += torch.sum((result_in > 0.5)).cpu().numpy()
                correct += torch.sum((result_out < 0.5)).cpu().numpy()
                print(
                    "acc : {:.2f}".format(
                        correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                target.to(self.device)
                loader = DataLoader(trainset(X_in, None, transform), batch_size=64, shuffle=False)
                output_in = forward(target, loader, self.device)
                output_in = torch.sort(output_in, dim=-1)[0][:, -self.topx:]
                loader = DataLoader(trainset(X_out, None, transform), batch_size=64, shuffle=False)
                output_out = forward(target, loader, self.device)
                output_out = torch.sort(output_out, dim=-1)[0][:, -self.topx:]
                _, _, result_in = self(F.softmax(output_in, dim=-1))
                _, _, result_out = self(F.softmax(output_out, dim=-1))
                correct = 0
                correct += torch.sum((result_in > 0.5)).cpu().numpy()
                correct += torch.sum((result_out < 0.5)).cpu().numpy()
                print(
                    "acc : {:.2f}".format(
                        correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
        else:
            if self.topx == -1:
                correct = 0
                for i in range(self.n_classes):
                    train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                         self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
                    train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                         torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(
                        self.device)
                    attack_model = self.attack_models[i]
                    attack_model.to(self.device)
                    loader = DataLoader(trainset(train_x, train_y), batch_size=64, shuffle=False)
                    for data in loader:
                        with torch.no_grad():
                            correct += torch.sum((attack_model(data[0]) > 0.5).float() == data[1]).cpu().numpy()
                print(
                    "acc : {:.2f}".format(
                        correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
            else:
                correct = 0
                train_x = torch.sort(torch.cat((self.shadowdata.data_in, self.shadowdata.data_out), dim=0), dim=-1)[0][
                          :, -self.topx:].to(self.device)
                train_y = torch.cat(
                    (torch.ones(self.shadowdata.data_in.shape[0]), torch.zeros(self.shadowdata.data_out.shape[0]))).to(
                    self.device)
                attack_model = self.attack_models[0]
                attack_model.to(self.device)
                loader = DataLoader(trainset(train_x, train_y), batch_size=64, shuffle=False)
                for data in loader:
                    with torch.no_grad():
                        correct += torch.sum((attack_model(data[0]) > 0.5).float() == data[1]).cpu().numpy()
                print(
                    "acc : {:.2f}".format(
                        correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))


class BoundaryDistance():
    def __init__(self, shadowmodel: ShadowModels, device: torch.device):
        self.shadowmodel = shadowmodel
        self.device = device
        self.max_samples = 5000
        self.acc_thresh = 0
        self.pre_thresh = 0

    def train(self):
        if not os.path.exists("./dist_shadow_in") or not os.path.exists("./dist_shadow_out"):
            dist_shadow_in = self.train_base(self.shadowmodel.loader_train, self.shadowmodel[0], self.max_samples,
                                             self.device)
            dist_shadow_out = self.train_base(self.shadowmodel.loader_test, self.shadowmodel[0], self.max_samples,
                                              self.device)
            pickle.dump(dist_shadow_in, open("./dist_shadow_in", "wb"))
            pickle.dump(dist_shadow_out, open("./dist_shadow_out", "wb"))
        else:
            dist_shadow_in = pickle.load(open("./dist_shadow_in", "rb"))
            dist_shadow_out = pickle.load(open("./dist_shadow_out", "rb"))
        dist_shadow = np.concatenate((dist_shadow_in, dist_shadow_out))
        membership_shadow = np.concatenate((np.ones_like(dist_shadow_in), np.zeros_like(dist_shadow_out)))
        acc, self.acc_thresh, prec, self.pre_thresh = get_threshold(membership_shadow, dist_shadow)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))

    @staticmethod
    def train_base(loader, model, max_samples, device):
        dist_adv = []
        num_samples = 0
        model.to(device)
        for i, data in enumerate(loader):
            # todo 分类错误为0
            xbatch, ybatch = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                y_pred = F.softmax(model(xbatch), dim=-1)
            x_selected = xbatch[torch.argmax(y_pred, dim=-1) == ybatch, :]
            x_adv_curr = hop_skip_jump_attack(model, x_selected, 2)
            d = torch.sqrt(torch.sum(torch.square(x_adv_curr - x_selected), dim=(1, 2, 3))).cpu().numpy()
            dist_adv.extend(d)
            num_samples += len(xbatch)
            if num_samples > max_samples:
                break
        return dist_adv[:max_samples]

    def __call__(self, model, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        x_adv_curr = hop_skip_jump_attack(model, X, 2)
        d = torch.sqrt(torch.sum(torch.square(x_adv_curr - X), dim=(1, 2, 3))).cpu().numpy()
        return X, Y, d > self.acc_thresh

    def evaluate(self, target, loader_in, loader_out):
        dist_target_in = self.train_base(loader_in, target, self.max_samples, self.device)
        dist_target_out = self.train_base(loader_out, target, self.max_samples, self.device)
        dist_target = np.concatenate((dist_target_in, dist_target_out))
        membership_target = np.concatenate((np.ones_like(dist_target_in), np.zeros_like(dist_target_out)))
        pickle.dump(dist_target_in, open("./dist_target_in", "wb"))
        pickle.dump(dist_target_out, open("./dist_target_out", "wb"))
        acc, _, _, _ = get_threshold(membership_target, dist_target, self.acc_thresh)
        _, _, prec, _ = get_threshold(membership_target, dist_target, self.pre_thresh)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))


class Augmentation():
    def __init__(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def __call__(self):
        pass
