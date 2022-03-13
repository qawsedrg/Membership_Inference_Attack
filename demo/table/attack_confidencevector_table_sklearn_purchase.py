import argparse
import os.path
import pickle

import numpy as np
import torch
from MIA.AttackModels import ConfidenceVector
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from MIA.ShadowModels import ShadowModels
from MIA.utils import trainset
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='purchase', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=1, type=int)
parser.add_argument("--attack_nepoch", default=5, type=int)
parser.add_argument("--topx", default=-1, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # net = KNeighborsClassifier(n_jobs=-1)
    net = Model(600)
    net.to(device)

    target = Model(600)
    target.to(device)
    target.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))
    target = KNeighborsClassifier(n_jobs=-1)

    # todo: should be repalced by a hill-climbing and GAN
    with open("../data/purchase_x", "rb") as f:
        X = pickle.load(f).astype(np.float32)
    with open("../data/purchase_y", "rb") as f:
        Y = pickle.load(f).astype(np.longlong)
    Y = np.squeeze(Y)
    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)

    target.fit(target_X, target_Y)

    shadow_models = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, args.shadow_nepoch, device)
    shadow_models.train()

    attack_model = ConfidenceVector(shadow_models, args.attack_nepoch, device, args.topx)
    attack_model.train()
    attack_model.evaluate()
    attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))

    data = target.predict_proba(target_X)
    if args.topx != -1:
        data = np.sort(data, axis=-1)[0][:, -args.topx:]
    loader = DataLoader(trainset(data), batch_size=1024, shuffle=False)
    membership = torch.Tensor().to(device)
    confidence_vectors = torch.Tensor().to(device)
    with torch.no_grad():
        for data in loader:
            data = data.to(device).float()
            result = attack_model(data)
            membership = torch.cat((membership, result[2]), dim=0)
            confidence_vectors = torch.cat((confidence_vectors, result[0]), dim=0)
    print("fini")
