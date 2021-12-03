import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from MIA.utils import trainset, train, attackmodel


class ConfidenceVector():
    def __init__(self, shadowmodel, epoches, device, topx=-1):
        self.shadowdata = shadowmodel.data
        #
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
                train_y = torch.cat((torch.ones_like(self.shadowdata.data_in[self.shadowdata.target_in == i]),
                                     torch.zeros_like(self.shadowdata.data_out[self.shadowdata.target_out == i])),
                                    dim=0)
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
            result=torch.cat((result,F.softmax(self.attack_models[i](x))),dim=0)
            data = torch.cat((data,x), dim=0)
        return data,result

    def evaluate(self):
        pass
