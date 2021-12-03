import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from MIA.AttackModels import ConfidenceVector
from MIA.ShadowModels import ShadowModels
from MIA.utils import trainset
from model import CIFAR

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

    attack_model = ConfidenceVector(shadow_models, 10, device, -1)
    attack_model.train()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    loader = DataLoader(trainset(X, transform=transform), batch_size=1024, shuffle=False)
    membership=torch.Tensor().to(device)
    confidence_vectors = torch.Tensor().to(device)
    for data in loader:
        data=data.to(device)
        data=F.softmax(net(data),dim=-1)
        result=attack_model(data)
        membership = torch.cat((membership, result[1]), dim=0)
        confidence_vectors = torch.cat((confidence_vectors, result[0]), dim=0)
    print("fini")
