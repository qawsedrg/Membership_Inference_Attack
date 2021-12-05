from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
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
        if self.topx == -1:
            self.attack_models = []
            for i in range(self.n_classes):
                train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                     self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
                train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                     torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(
                    self.device)
                attack_model = attackmodel(train_x.shape[-1])
                optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
                loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
                print("Training attack model for class {:}".format(i))
                attack_model = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                     epoches=self.epoches)
                self.attack_models.append(attack_model)
        else:
            pass

    def __call__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
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

    def evaluate(self, target: Optional[nn.Module] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None):
        if target is not None:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            target.to(self.device)
            loader = DataLoader(trainset(X_in, None, transform), batch_size=64, shuffle=True)
            output_in = forward(target, loader, self.device)
            loader = DataLoader(trainset(X_out, None, transform), batch_size=64, shuffle=True)
            output_out = forward(target, loader, self.device)
            _, _, result_in = self(output_in)
            _, _, result_out = self(output_out)
            correct = 0
            correct += torch.sum((result_in > 0.5)).cpu().numpy()
            correct += torch.sum((result_out < 0.5)).cpu().numpy()
            print(
                "acc : {:.2f}".format(correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
        else:
            correct = 0
            for i in range(self.n_classes):
                train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                     self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
                train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                     torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(
                    self.device)
                attack_model = self.attack_models[i]
                loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
                for data in loader:
                    with torch.no_grad():
                        correct += torch.sum((attack_model(data[0]) > 0.5).float() == data[1]).cpu().numpy()
            print(
                "acc : {:.2f}".format(correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
