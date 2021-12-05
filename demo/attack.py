import argparse
import os.path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from MIA.AttackModels import ConfidenceVector
from MIA.ShadowModels import ShadowModels
from MIA.utils import trainset
from model import CIFAR

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='cifar10', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=10, type=int)
parser.add_argument("--attack_nepoch", default=5, type=int)
parser.add_argument("--topx", default=3, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CIFAR(10)
    net.to(device)

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

    shadow_models = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, args.shadow_nepoch, device)
    shadow_models.train()

    attack_model = ConfidenceVector(shadow_models, args.attack_nepoch, device, args.topx)
    attack_model.train()
    attack_model.evaluate()
    attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    loader = DataLoader(trainset(target_X, transform=transform), batch_size=1024, shuffle=False)
    membership = torch.Tensor().to(device)
    confidence_vectors = torch.Tensor().to(device)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data = F.softmax(net(data), dim=-1)
            if args.topx != -1:
                data = torch.sort(data, dim=-1)[0][:, -args.topx:]
            result = attack_model(data)
            membership = torch.cat((membership, result[2]), dim=0)
            confidence_vectors = torch.cat((confidence_vectors, result[0]), dim=0)
    print("fini")
