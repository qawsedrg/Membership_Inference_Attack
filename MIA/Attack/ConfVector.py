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
        r"""
        Confidence Vector Attack model

        Direct classification of confidence vectors

        :param shadowmodel: shadowmodel
        :param epoches: epoches to train the attack model
        :param device: torch.device object
        :param topx: -1 (whole vector) or any positive integer less than the length of confidence vector
        :param transform: transformation to perform on images
        """
        self.shadowdata = shadowmodel.data
        self.collate_fn = shadowmodel.collate_fn
        self.n_classes = int(max(torch.max(self.shadowdata.target_in).cpu().numpy(),
                                 torch.max(self.shadowdata.target_out).cpu().numpy()) + 1)
        self.topx = topx
        self.epoches = epoches
        self.device = device
        self.transform = transform

    def train(self) -> None:
        r"""
        Train the Confidence Vector Attack model

        .. note::
            If topx == -1, an attack model will be trained for each class

            If not, only one attack model will be trained
        """
        self.attack_models = []
        if self.topx == -1:
            # classification of whole vector, train an attack model per class
            def f(Is):
                attack_models = []
                for i in Is:
                    train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                         self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
                    # trained - 1, not trained - 0
                    train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                         torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(
                        self.device)
                    attack_model = attackmodel(train_x.shape[-1])
                    attack_model.to(self.device)
                    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
                    loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
                    print("Training attack model for class {:}".format(i))
                    attack_model, _, _ = train(attack_model, loader, self.device, optimizer=optimizer,
                                               criterion=nn.BCELoss(),
                                               epoches=self.epoches)
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
            # classification of sorted vector with topx elements, only one attack model is trained
            train_x = torch.sort(torch.cat((self.shadowdata.data_in, self.shadowdata.data_out), dim=0), dim=-1)[0][:,
                      -self.topx:].to(self.device)
            train_y = torch.cat(
                (torch.ones(self.shadowdata.data_in.shape[0]), torch.zeros(self.shadowdata.data_out.shape[0]))).to(
                self.device)
            attack_model = attackmodel(train_x.shape[-1])
            attack_model.to(self.device)
            optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
            loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
            attack_model, _, _ = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                       epoches=self.epoches)
            self.attack_models.append(attack_model)

    def __call__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Inference of membership

        .. note::
            The output membership is not in the same order as the input X but as the output data_x, data_y

        :param X: data to infer
        :param Y: class of data to infer
        :return: X, Y, membership (represented by probability)
        """
        if self.topx == -1:
            classes = torch.max(X, dim=-1).indices
            result = torch.Tensor().to(self.device)
            data_x = torch.Tensor().to(self.device)
            data_y = torch.Tensor().to(self.device)
            for i in range(self.n_classes):
                # classification of whole vector, an attack model per class is trained
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
        r"""
        Show the distribution of Ordered Confidence Vectors (3D)

        .. note::
            topx if forced to 3

        .. note::
            The Confidence Vectors should be longer than 3
        """
        if self.topx == -1:
            topx = 0
        else:
            topx = 3
        if self.shadowdata.data_in.shape[-1] < 3:
            return
        data_in = torch.sort(self.shadowdata.data_in, dim=-1)[0][:, -topx:].cpu()
        data_out = torch.sort(self.shadowdata.data_out, dim=-1)[0][:, -topx:].cpu()

        ax = plt.axes(projection='3d')

        ax.scatter3D(data_out[:, -3], data_out[:, -2], data_out[:, -1], marker='^', label="Not Trained")
        ax.scatter3D(data_in[:, -3], data_in[:, -2], data_in[:, -1], marker='o', label="Trained")

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()

        plt.show()

    def evaluate(self, target: Optional[Union[BaseEstimator, nn.Module]] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None) -> Tuple[float, float]:
        r"""
        Evaluate on shadowmodel or victime model

        Example::
            attack_model.evaluate(target, *train_test_split(X, Y, test_size=.5)) # on victime model

            attack_model.evaluate() # on shadowmodel

        :param target: victime sklearn model or torch model
        :param X_in: data trained
        :param X_out: data not trianed
        :param Y_in: class of data trained
        :param Y_out: class of data not trained
        :return: accuracy, precision
        """
        # evaluate on shadowmodel
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
                output_in = torch.sort(output_in)[0][:, -self.topx:]
                output_out = torch.sort(output_out)[0][:, -self.topx:]
            # should be changed if softmax is performed in the model
            _, _, result_in = self(F.softmax(output_in, dim=-1) if
                                   isinstance(target, nn.Module) else output_in)
            _, _, result_out = self(F.softmax(output_out, dim=-1) if isinstance(target, nn.Module) else output_out)
            correct = 0
            correct += torch.sum((result_in > 0.5)).cpu().numpy()
            prec = correct / (X_in.shape[0])
            correct += torch.sum((result_out < 0.5)).cpu().numpy()
            acc = correct / (X_in.shape[0] + X_out.shape[0])
            print(
                "acc : {:.2f}, prec : {:.2f}".format(acc, prec))
            return acc, prec
        # evaluate on victime model
        else:
            if self.topx == -1:
                correct = 0
                prec = 0
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
                            correct += torch.sum(((attack_model(data[0]) > 0.5).int() == data[1])).cpu().numpy()
                            prec += torch.sum(
                                ((attack_model(data[0]) > 0.5).int() == data[1]) * (data[1] == 1)).cpu().numpy()
                acc = correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])
                prec /= self.shadowdata.data_in.shape[0]
                print(
                    "acc : {:.2f}, prec : {:.2f}".format(acc, prec))
                return acc, prec
            else:
                correct = 0
                prec = 0
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
                        correct += torch.sum(((attack_model(data[0]) > 0.5).int() == data[1])).cpu().numpy()
                        prec += torch.sum(
                            ((attack_model(data[0]) > 0.5).int() == data[1]) * (data[1] == 1)).cpu().numpy()
                acc = correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])
                prec /= self.shadowdata.data_in.shape[0]
                print(
                    "acc : {:.2f}, prec : {:.2f}".format(acc, prec))
                return acc, prec
