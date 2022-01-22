import argparse
import os.path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from MIA.torch.AttackModels import NoiseAttack
from MIA.torch.utils import trainset
from MIA.torch.ShadowModels import ShadowModels
from model import CIFAR

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='cifar100', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=15, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CIFAR(100)
    net.to(device)

    target = CIFAR(100)
    target.to(device)
    target.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))

    # todo: should be repalced by a hill-climbing and GAN
    train = torchvision.datasets.CIFAR100(root='../data', train=True,
                                          download=True)
    test = torchvision.datasets.CIFAR100(root='../data', train=False,
                                         download=True)
    X, Y = np.concatenate((train.data, test.data)), np.concatenate((train.targets, test.targets)).astype(np.int64)
    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)

    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    shadow_models = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, args.shadow_nepoch, device,
                                 transform)
    shadow_models.train()

    attack_model = NoiseAttack(shadow_models, device, transform)
    attack_model.train(show=True)
    attack_model.evaluate(target,
                          *train_test_split(target_X[:5000, :], target_Y[:5000], test_size=0.5, random_state=42))

    loader = DataLoader(trainset(target_X[:1000, :], transform=transform), batch_size=100, shuffle=False)
    membership = np.array([])
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            result = attack_model(target, data)
            membership = np.concatenate((membership, result), axis=0)
    print("fini")
