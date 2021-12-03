import argparse
import torch
from model import CIFAR
import torchvision
from MIA.ShadowModels import *
from MIA.AttackModels import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", default=30, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='cifar10', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CIFAR()
    net.to(device)

    # todo: should be repalced by a hill-climbing and GAN
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True)
    X, Y = testset.data, testset.targets

    shadow_models = ShadowModels(net, 2, X, Y, 1, device)
    shadow_models.train()

    attack_model = ConfidenceVector(shadow_models,10,device,-1)
    attack_model.train()

    loader = DataLoader(Dataset(X), batch_size=64, shuffle=False)
    predicted=torch.tensor().to(device)
    for i,x in enumerate(loader):
        x=x.to(device)
        predicted=torch.cat((attack_model(F.softmax(net(x))),predicted),dim=0)