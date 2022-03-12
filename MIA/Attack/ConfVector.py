import multiprocessing
from multiprocessing.pool import ThreadPool
from typing import Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import DataLoader

from MIA import ShadowModels
from MIA.utils import trainset, train, attackmodel, forward, forward_sklearn


class ConfVector():
    def __init__(self, shadowmodel: ShadowModels, epoches: int, device: torch.device, topx: Optional[int] = -1,
                 transform: Optional = None):
        self.shadowdata = shadowmodel.data
        self.collate_fn = shadowmodel.collate_fn
        self.n_classes = int(max(torch.max(self.shadowdata.target_in).cpu().numpy(),
                                 torch.max(self.shadowdata.target_out).cpu().numpy()) + 1)
        self.topx = topx
        self.epoches = epoches
        self.device = device
        self.transform = transform

    def train(self) -> None:
        self.attack_models = []
        if self.topx == -1:
            def f(Is):
                attack_models = []
                for i in Is:
                    train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                         self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
                    train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                         torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(
                        self.device)
                    attack_model = attackmodel(train_x.shape[-1])
                    attack_model.to(self.device)
                    optimizer = optim.Adam(attack_model.parameters(), lr=0.01)
                    loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
                    print("Training attack model for class {:}".format(i))
                    attack_model = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                         epoches=self.epoches, verbose=False)
                    attack_models.append(attack_model)
                return attack_models

            numberOfThreads = min(multiprocessing.cpu_count(), self.n_classes)
            pool = ThreadPool(processes=numberOfThreads)
            Chunks = np.array_split(list(range(self.n_classes)), numberOfThreads)
            results = pool.map_async(f, Chunks)
            pool.close()
            pool.join()
            for result in results.get():
                attack_model = result
                self.attack_models.extend(attack_model)

        else:
            train_x = torch.sort(torch.cat((self.shadowdata.data_in, self.shadowdata.data_out), dim=0), dim=-1)[0][:,
                      -self.topx:].to(self.device)
            train_y = torch.cat(
                (torch.ones(self.shadowdata.data_in.shape[0]), torch.zeros(self.shadowdata.data_out.shape[0]))).to(
                self.device)
            attack_model = attackmodel(train_x.shape[-1])
            attack_model.to(self.device)
            optimizer = optim.Adam(attack_model.parameters(), lr=0.01)
            loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
            attack_model = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                 epoches=self.epoches)
            self.attack_models.append(attack_model)

    def __call__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def show(self) -> None:
        if self.topx == -1:
            topx = 0
        else:
            topx = 3
        if self.shadowdata.data_in.shape[-1] < 3:
            return
        data_in = torch.sort(self.shadowdata.data_in, dim=-1)[0][:, -topx:].cpu()
        data_out = torch.sort(self.shadowdata.data_out, dim=-1)[0][:, -topx:].cpu()

        ax = plt.axes(projection='3d')

        ax.scatter3D(data_out[:, 0], data_out[:, 1], data_out[:, 2], marker='^', label="Not Trained")
        ax.scatter3D(data_in[:, 0], data_in[:, 1], data_in[:, 2], marker='o', label="Trained")

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()

        plt.show()

    def evaluate(self, target: Optional[Union[BaseEstimator, nn.Module]] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None) -> float:
        """

        :param target:
        :param X_in:
        :param X_out:
        :param Y_in:
        :param Y_out:
        :return:
        """
        if target is not None:
            loader = DataLoader(trainset(X_in, Y_in, self.transform), batch_size=64, shuffle=False,
                                collate_fn=self.collate_fn)
            output_in = forward(target, loader, self.device) if isinstance(target, nn.Module) else forward_sklearn(
                target, loader, self.device)
            loader = DataLoader(trainset(X_out, Y_out, self.transform), batch_size=64, shuffle=False,
                                collate_fn=self.collate_fn)
            output_out = forward(target, loader, self.device) if isinstance(target, nn.Module) else forward_sklearn(
                target, loader, self.device)
            if self.topx != -1:
                output_in = output_in[0][:, -self.topx:]
                output_out = output_out[0][:, -self.topx:]
            _, _, result_in = self(F.softmax(output_in, dim=-1) if isinstance(target, nn.Module) else output_in)
            _, _, result_out = self(F.softmax(output_out, dim=-1) if isinstance(target, nn.Module) else output_out)
            correct = 0
            correct += torch.sum((result_in > 0.5)).cpu().numpy()
            correct += torch.sum((result_out < 0.5)).cpu().numpy()
            acc = correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])
            print(
                "acc : {:.2f}".format(acc))
            return acc
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
                acc = correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])
                print(
                    "acc : {:.2f}".format(acc))
                return acc
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
                acc = correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])
                print(
                    "acc : {:.2f}".format(acc))
                return acc
