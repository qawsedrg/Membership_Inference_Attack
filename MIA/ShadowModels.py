import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from utils import *


class ShadowModels:
    def __init__(self, models, N: int, X, Y, epoches, device):
        self.models = models
        self.N = N
        self.X = X
        self.Y = Y
        self.epoches = epoches
        self.device = device
        self.shadowmodels = []

    def train(self):
        for i in range(self.N):
            model = self.models
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            shadow_X_train, shadow_X_test, shadow_Y_train, shadow_Y_test = train_test_split(self.X, self.Y,
                                                                                            test_size=0.5,
                                                                                            random_state=i)
            loader = trainset(shadow_X_train, shadow_Y_train)
            model = train(model, loader, self.device, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
                          epoches=self.epoches)
            self.shadowmodels.append(model)
