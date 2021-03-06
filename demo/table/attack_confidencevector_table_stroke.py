import argparse
import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from MIA.Attack.ConfVector import ConfVector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from MIA.ShadowModels import ShadowModels
from MIA.utils import trainset
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='purchase', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=50, type=int)
parser.add_argument("--attack_nepoch", default=5, type=int)
parser.add_argument("--topx", default=-1, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Model(10)
    net.to(device)

    target = Model(10)
    target.to(device)
    target.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))

    data = pd.read_csv('../data/stroke.csv', nrows=600)
    data = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']].dropna()
    le = LabelEncoder()
    for i in data.columns:
        data[i] = le.fit_transform(data[i])
    print(data)

    X = np.array(data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                       'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]).astype(np.float32)
    Y = np.array(data['stroke']).astype(np.longlong)
    Y = np.squeeze(Y)

    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)

    shadow_models = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, args.shadow_nepoch, device)
    shadow_models.train()

    attack_model = ConfVector(shadow_models, args.attack_nepoch, device, args.topx)
    attack_model.train()
    attack_model.evaluate()
    attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))

    loader = DataLoader(trainset(target_X), batch_size=1024, shuffle=False)
    membership = torch.Tensor().to(device)
    confidence_vectors = torch.Tensor().to(device)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data = F.softmax(target(data), dim=-1)
            if args.topx != -1:
                data = torch.sort(data, dim=-1)[0][:, -args.topx:]
            result = attack_model(data)
            membership = torch.cat((membership, result[2]), dim=0)
            confidence_vectors = torch.cat((confidence_vectors, result[0]), dim=0)
    print("fini")
