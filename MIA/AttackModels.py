import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from MIA.utils import trainset, train, attackmodel


class ConfidenceVector():
    def __init__(self, shadowmodel, epoches, device, topx=-1):
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
                attack_model = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                     epoches=self.epoches)
                self.attack_models.append(attack_model)
        else:
            pass

    def __call__(self, X):
        classes = torch.max(X, dim=-1).indices
        result = torch.Tensor().to(self.device)
        data = torch.Tensor().to(self.device)
        for i in range(self.n_classes):
            x = X[classes == i]
            result = torch.cat((result, self.attack_models[i](x)))
            data = torch.cat((data, x), dim=0)
        return data, result

    def evaluate(self):
        correct = 0
        for i in range(self.n_classes):
            train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                 self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
            train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                 torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(self.device)
            attack_model = self.attack_models[i]
            loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
            for data in loader:
                correct += torch.sum((attack_model(data[0]) > 0.5).float() == data[1]).cpu().numpy()
        print("acc : {:.2f}".format(correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
