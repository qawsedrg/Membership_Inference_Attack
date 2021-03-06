from copy import deepcopy
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
                 transform: Optional = None, collate_fn: Optional = None, opt: Optional = None, lr: Optional = None,
                 eval: Optional[bool] = True):
        r"""
        Training of shadowmodels, compute the output models which can be indexed and sliced, confidence vecteurs and ground truth label associated, return accuray

        :param models: structure used to train the shadowmodels
        :param N: number of shadowmodel trained
        :param X: data
        :param Y: label of data
        :param epoches: epoches
        :param device: torch.device object
        :param transform: transformation to perform on images
        :param collate_fn: collate_fn used in DataLoader
        :param opt: optimizer function
        :param lr: learning rate
        :param eval: evaluate the shadowmodels and infer the confidence vectors
        """
        # TODO: multiple models with different structure
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
        self.eval = eval

    def train(self):
        X_in = torch.Tensor().to(self.device)
        Y_in = torch.Tensor().to(self.device)
        X_out = torch.Tensor().to(self.device)
        Y_out = torch.Tensor().to(self.device)
        acc_list = []
        val_acc_list = []
        if isinstance(self.models, ShadowModels):
            # reuse the trained shadowmodels
            for i in range(self.N):
                model = self.models[i]
                shadow_X_train, shadow_X_test, shadow_Y_train, shadow_Y_test = train_test_split(self.X, self.Y,
                                                                                                test_size=0.5,
                                                                                                random_state=i)

                loader_train = DataLoader(trainset(shadow_X_train, shadow_Y_train, self.transform), batch_size=64,
                                          shuffle=False, collate_fn=self.collate_fn)
                loader_test = DataLoader(trainset(shadow_X_test, shadow_Y_test, self.transform), batch_size=64,
                                         shuffle=False, collate_fn=self.collate_fn)
                model, acc, val_acc = train(model, self.device, testloader=loader_test, eval=True, train=False)
                model.eval()
                # hard coded
                # used only by boundary and noise attack model
                # but they only use one shadowmodel
                self.loader_train = loader_train
                self.loader_test = loader_test
                # in: trained, out: not trained
                # should be changed if softmax is performed in the model
                X_in = torch.cat((X_in, F.softmax(forward(model, loader_train, self.device), dim=-1)), dim=0)
                X_out = torch.cat((X_out, F.softmax(forward(model, loader_test, self.device), dim=-1)), dim=0)
                Y_in = torch.cat((Y_in, torch.from_numpy(np.array(shadow_Y_train)).to(self.device)), dim=0)
                Y_out = torch.cat((Y_out, torch.from_numpy(np.array(shadow_Y_test)).to(self.device)), dim=0)
                # Y_in = torch.cat((Y_in, torch.argmax(X_in, dim=-1)), dim=0)
                # Y_out = torch.cat((Y_out, torch.argmax(X_out, dim=-1)), dim=0)
                acc_list.append(acc)
                val_acc_list.append(val_acc)
        elif isinstance(self.models, nn.Module):
            # torch model
            for i in range(self.N):
                # copy the original structure and param of model to be trained
                # ifnot the model will be trained on the already trained model
                model = deepcopy(self.models)
                optimizer = self.opt(model.parameters(), lr=self.lr)
                shadow_X_train, shadow_X_test, shadow_Y_train, shadow_Y_test = train_test_split(self.X, self.Y,
                                                                                                test_size=0.5,
                                                                                                random_state=i)
                loader = DataLoader(trainset(shadow_X_train, shadow_Y_train, self.transform), batch_size=64,
                                    shuffle=True,
                                    collate_fn=self.collate_fn)
                testloader = DataLoader(trainset(shadow_X_test, shadow_Y_test, self.transform), batch_size=64,
                                        shuffle=False,
                                        collate_fn=self.collate_fn)
                model, acc, val_acc = train(model, self.device, loader=loader, optimizer=optimizer,
                                            criterion=nn.CrossEntropyLoss(),
                                            epoches=self.epoches, testloader=testloader, eval=self.eval)
                self.model_trained.append(model)
                if self.eval:
                    model.eval()
                    loader_train = DataLoader(trainset(shadow_X_train, shadow_Y_train, self.transform), batch_size=64,
                                              shuffle=False, collate_fn=self.collate_fn)
                    loader_test = DataLoader(trainset(shadow_X_test, shadow_Y_test, self.transform), batch_size=64,
                                             shuffle=False, collate_fn=self.collate_fn)
                    # hard coded
                    # used only by boundary and noise attack model
                    # but they only use one shadowmodel
                    self.loader_train = loader_train
                    self.loader_test = loader_test
                    # in: trained, out: not trained
                    # should be changed if softmax is performed in the model
                    X_in = torch.cat((X_in, F.softmax(forward(model, loader_train, self.device), dim=-1)), dim=0)
                    X_out = torch.cat((X_out, F.softmax(forward(model, loader_test, self.device), dim=-1)), dim=0)
                    Y_in = torch.cat((Y_in, torch.from_numpy(np.array(shadow_Y_train)).to(self.device)), dim=0)
                    Y_out = torch.cat((Y_out, torch.from_numpy(np.array(shadow_Y_test)).to(self.device)), dim=0)
                    # Y_in = torch.cat((Y_in, torch.argmax(X_in, dim=-1)), dim=0)
                    # Y_out = torch.cat((Y_out, torch.argmax(X_out, dim=-1)), dim=0)
                    acc_list.append(acc)
                    val_acc_list.append(val_acc)
        else:
            # sklearn model
            model = self.models
            for i in range(self.N):
                shadow_X_train, shadow_X_test, shadow_Y_train, shadow_Y_test = train_test_split(self.X, self.Y,
                                                                                                test_size=0.5,
                                                                                                random_state=i)
                model = model.fit(shadow_X_train, shadow_Y_train)
                self.model_trained.append(model)
                X_in = torch.cat((X_in, torch.from_numpy(model.predict_proba(shadow_X_train)).to(self.device)), dim=0)
                X_out = torch.cat((X_out, torch.from_numpy(model.predict_proba(shadow_X_test)).to(self.device)), dim=0)
                Y_in = torch.cat((Y_in, torch.from_numpy(np.array(shadow_Y_train)).to(self.device)), dim=0)
                Y_out = torch.cat((Y_out, torch.from_numpy(np.array(shadow_Y_test)).to(self.device)), dim=0)

        self.data = DataStruct(X_in.float(), X_out.float(), Y_in, Y_out)
        return acc_list, val_acc_list

    def __getitem__(self, item):
        return self.model_trained[item]

    def modules(self):
        raise

    def parameters(self):
        raise
