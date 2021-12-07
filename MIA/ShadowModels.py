import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from MIA.utils import trainset, train, forward, DataStruct


class ShadowModels:
    def __init__(self, models: nn.Module, N: int, X: torch.Tensor, Y: torch.Tensor, epoches: int, device: torch.device):
        self.models = models
        self.N = N
        self.X = X
        self.Y = Y
        self.epoches = epoches
        self.device = device
        self.data = None

    def train(self):
        X_in = torch.Tensor().to(self.device)
        Y_in = torch.Tensor().to(self.device)
        X_out = torch.Tensor().to(self.device)
        Y_out = torch.Tensor().to(self.device)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i in range(self.N):
            model = self.models
            optimizer = optim.Adam(model.parameters(), lr=0.001)
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
                # Y_in = torch.cat((Y_in, torch.from_numpy(np.array(shadow_Y_train)).to(self.device)), dim=0)
                # Y_out = torch.cat((Y_out, torch.from_numpy(np.array(shadow_Y_test)).to(self.device)), dim=0)
                Y_in = torch.cat((Y_in, torch.argmax(X_in, dim=-1)), dim=0)
                Y_out = torch.cat((Y_out, torch.argmax(X_out, dim=-1)), dim=0)

        self.data = DataStruct(F.softmax(X_in, dim=-1), F.softmax(X_out, dim=-1), Y_in, Y_out)
