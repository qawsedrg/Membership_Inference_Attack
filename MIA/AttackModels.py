from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from torch import nn
from torch.utils.data import DataLoader

from MIA.ShadowModels import ShadowModels
from MIA.utils import trainset, train, attackmodel, forward


class ConfidenceVector():
    def __init__(self, shadowmodel: ShadowModels, epoches: int, device: torch.device, topx: Optional[int] = -1):
        self.shadowdata = shadowmodel.data
        self.n_classes = int(max(torch.max(self.shadowdata.target_in).cpu().numpy(),
                                 torch.max(self.shadowdata.target_out).cpu().numpy()) + 1)
        self.topx = topx
        self.epoches = epoches
        self.device = device

    def train(self):
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
    def __init__(self, shadowmodel: ShadowModels, epoches: int, device: torch.device, topx: Optional[int] = -1):
        self.shadowmodel = shadowmodel
        self.shadowdata = shadowmodel.data
        self.n_classes = int(max(torch.max(self.shadowdata.target_in).cpu().numpy(),
                                 torch.max(self.shadowdata.target_out).cpu().numpy()) + 1)
        self.topx = topx
        self.epoches = epoches
        self.device = device
        self.max_samples = 5000

    def train(self):
        dist_shadow_in = self.train_base(self.shadowmodel.loader_train, self.shadowmodel[0], self.max_samples,
                                         self.device)
        dist_shadow_in = self.train_base(self.shadowmodel.loader_test, self.shadowmodel[0], self.max_samples,
                                         self.device)

    @staticmethod
    def train_base(loader, model, max_samples, device):
        dist_adv = []
        num_samples = 0
        model.to(device)
        for i, data in enumerate(loader):
            xbatch, ybatch = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                y_pred = F.softmax(model(xbatch), dim=-1)
            x_selected = xbatch[torch.argmax(y_pred, dim=-1) == ybatch, :][:2, :]
            x_adv_curr = hop_skip_jump_attack(model, x_selected, 2)
            d = torch.sqrt(torch.sum(torch.square(x_adv_curr - x_selected), dim=(1, 2, 3))).cpu().numpy()
            dist_adv.extend(d)
            num_samples += len(xbatch)
            if num_samples > max_samples:
                break
        return dist_adv[:max_samples]
