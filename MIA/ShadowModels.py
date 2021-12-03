import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
from MIA.utils import *
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class ShadowModels:
    def __init__(self, models, N: int, X, Y, epoches, device):
        self.models = models
        self.N = N
        self.X = X
        self.Y = Y
        self.epoches = epoches
        self.device = device
        self.data=None

    def train(self):
        X_in = torch.Tensor().to(self.device)
        Y_in = torch.Tensor().to(self.device)
        X_out = torch.Tensor().to(self.device)
        Y_out = torch.Tensor().to(self.device)
        for i in range(self.N):
            model = self.models
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            shadow_X_train, shadow_X_test, shadow_Y_train, shadow_Y_test = train_test_split(self.X, self.Y,
                                                                                            test_size=0.5,
                                                                                            random_state=i)
            loader = DataLoader(trainset(shadow_X_train, shadow_Y_train, transform), batch_size=64, shuffle=True)
            model = train(model, loader, self.device, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
                          epoches=self.epoches)
            model.eval()
            with torch.no_grad():
                loader_train = DataLoader(trainset(shadow_X_train, shadow_Y_train, transform), batch_size=64,
                                          shuffle=False)
                loader_test = DataLoader(trainset(shadow_X_test, shadow_Y_test, transform), batch_size=64,
                                         shuffle=False)
                X_in = torch.cat((X_in, forward(model, loader_train, self.device)), dim=0)
                X_out = torch.cat((X_out, forward(model, loader_test, self.device)), dim=0)
                Y_in = torch.cat((Y_in, torch.from_numpy(np.array(shadow_Y_train)).to(self.device)), dim=0)
                Y_out = torch.cat((Y_out, torch.from_numpy(np.array(shadow_Y_test)).to(self.device)), dim=0)

        self.data=DataStruct(X_in, X_out, Y_in, Y_out)
