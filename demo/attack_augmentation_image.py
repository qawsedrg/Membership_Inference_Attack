import argparse
import os.path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from MIA.AttackModels import Augmentation
from model import CIFAR

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='cifar10', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=15, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    target = CIFAR(10)
    target.to(device)
    target.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))

    # todo: should be repalced by a hill-climbing and GAN
    train = torchvision.datasets.CIFAR10(root='../data', train=True,
                                         download=True)
    test = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True)
    X, Y = np.concatenate((train.data, test.data)), np.concatenate((train.targets, test.targets)).astype(np.int64)
    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)

    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    attack_model = Augmentation(device, transform=transform)
    attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42), show=True)
    membership = attack_model(target, target_X, target_Y)