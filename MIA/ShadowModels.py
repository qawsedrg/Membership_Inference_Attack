from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from MIA.utils import trainset, train, forward, DataStruct


class ShadowModels:
    def __init__(self, models, N: int, X: torch.Tensor, Y: torch.Tensor, epoches: int, device: torch.device,
                 transform: Optional = None, collate_fn: Optional = None, opt: Optional = None, lr: Optional = None):
        self.models = models
        self.N = N
        self.X = X
        self.Y = Y
        self.epoches = epoches
        self.device = device
        self.data = None
        self.model_trained = []
        self.loader_train = None
        self.loader_test = None
        self.transform = transform
        self.collate_fn = collate_fn
        self.opt = opt if opt != None else optim.Adam
        self.lr = lr if lr != None else 0.001
        self.gap = []

    def train(self):
        X_in = torch.Tensor().to(self.device)
        Y_in = torch.Tensor().to(self.device)
        X_out = torch.Tensor().to(self.device)
        Y_out = torch.Tensor().to(self.device)
        acc = 0;
        val_acc = 0
        for i in range(self.N):
            model = self.models
            shadow_X_train, shadow_X_test, shadow_Y_train, shadow_Y_test = train_test_split(self.X, self.Y,
                                                                                            test_size=0.5,
                                                                                            random_state=i)
            if isinstance(model, nn.Module):
                optimizer = self.opt(model.parameters(), lr=self.lr)
                for idx, module in enumerate(model.modules()):
                    if idx != 0:
                        try:
                            module.reset_parameters()
                        except:
                            pass
                loader = DataLoader(trainset(shadow_X_train, shadow_Y_train, self.transform), batch_size=64,
                                    shuffle=True,
                                    collate_fn=self.collate_fn)
                testloader = DataLoader(trainset(shadow_X_test, shadow_Y_test, self.transform), batch_size=64,
                                        shuffle=False,
                                        collate_fn=self.collate_fn)
                model, acc, val_acc = train(model, loader, self.device, optimizer=optimizer,
                                            criterion=nn.CrossEntropyLoss(),
                                            epoches=self.epoches, testloader=testloader, eval=True)
                self.model_trained.append(model)
                self.gap.append((acc, val_acc))
                model.eval()
                loader_train = DataLoader(trainset(shadow_X_train, shadow_Y_train, self.transform), batch_size=64,
                                          shuffle=False, collate_fn=self.collate_fn)
                loader_test = DataLoader(trainset(shadow_X_test, shadow_Y_test, self.transform), batch_size=64,
                                         shuffle=False, collate_fn=self.collate_fn)
                self.loader_train = loader_train
                self.loader_test = loader_test
                X_in = torch.cat((X_in, F.softmax(forward(model, loader_train, self.device), dim=-1)), dim=0)
                X_out = torch.cat((X_out, F.softmax(forward(model, loader_test, self.device), dim=-1)), dim=0)
                Y_in = torch.cat((Y_in, torch.from_numpy(np.array(shadow_Y_train)).to(self.device)), dim=0)
                Y_out = torch.cat((Y_out, torch.from_numpy(np.array(shadow_Y_test)).to(self.device)), dim=0)
                # Y_in = torch.cat((Y_in, torch.argmax(X_in, dim=-1)), dim=0)
                # Y_out = torch.cat((Y_out, torch.argmax(X_out, dim=-1)), dim=0)
            else:
                model = model.fit(shadow_X_train, shadow_Y_train)
                self.model_trained.append(model)
                X_in = torch.cat((X_in, torch.from_numpy(model.predict_proba(shadow_X_train)).to(self.device)), dim=0)
                X_out = torch.cat((X_out, torch.from_numpy(model.predict_proba(shadow_X_test)).to(self.device)), dim=0)
                Y_in = torch.cat((Y_in, torch.from_numpy(np.array(shadow_Y_train)).to(self.device)), dim=0)
                Y_out = torch.cat((Y_out, torch.from_numpy(np.array(shadow_Y_test)).to(self.device)), dim=0)

        self.data = DataStruct(X_in.float(), X_out.float(), Y_in, Y_out)
        return acc, val_acc

    def __getitem__(self, item):
        return self.model_trained[item]
