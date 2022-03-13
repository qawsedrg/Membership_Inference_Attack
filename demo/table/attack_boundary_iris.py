import argparse
import os.path

import numpy as np
import torch
from MIA.AttackModels import BoundaryDistance
from sklearn import datasets
from sklearn.model_selection import train_test_split

from MIA.ShadowModels import ShadowModels
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='iris', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=50, type=int)
parser.add_argument("--attack_nepoch", default=5, type=int)
parser.add_argument("--topx", default=-1, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Model(4, 3)
    net.to(device)

    target = Model(4, 3)
    target.to(device)
    target.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))

    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)
    Y = iris.target.astype(np.longlong)

    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)

    shadow_models = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, args.shadow_nepoch, device)
    shadow_models.train()

    attack_model = BoundaryDistance(shadow_models, device)
    attack_model.train(show=True)
    attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))

    print("fini")
