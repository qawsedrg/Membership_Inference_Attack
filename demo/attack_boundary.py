import argparse
import os.path

import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split

from MIA.AttackModels import BoundaryDistance
from MIA.ShadowModels import ShadowModels
from model import CIFAR

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='cifar100', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=10, type=int)
parser.add_argument("--attack_nepoch", default=1, type=int)
parser.add_argument("--topx", default=-1, type=int)

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

    shadow_models = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, args.shadow_nepoch, device)
    shadow_models.train()

    attack_model = BoundaryDistance(shadow_models, args.attack_nepoch, device, args.topx)
    attack_model.train()
