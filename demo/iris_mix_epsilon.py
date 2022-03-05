import pandas as pd
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os.path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from MIA.AttackModels import ConfidenceVector
from MIA.ShadowModels import ShadowModels
from model import Model
from sklearn import datasets
from MIA.utils import trainset, mix

e = np.linspace(1, 5, 100)
b = []
c = []
n = 10

for epsilon in e:
    b0 = 0
    c0 = 0
    for aa in range(n):
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", default=50, type=int)
        parser.add_argument("--batch_size", default=20, type=int)
        parser.add_argument("--save_to", default='models', type=str)
        parser.add_argument("--name", default='iris', type=str, choices=["purchase", "location", "adult"], )
        parser.add_argument('--decay', default=1e-2, type=float, help='weight decay (default=1e-2)')


        args = parser.parse_args()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        iris = datasets.load_iris()
        X = iris.data.astype(np.float32)
        Y = iris.target.astype(np.longlong)

        target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)
        target_train_X, target_test_X, target_train_Y, target_test_Y = train_test_split(target_X, target_Y,
                                                                                        test_size=0.5,
                                                                                        random_state=42)
        shadow_train_X, shadow_test_X, shadow_train_Y, shadow_test_Y = train_test_split(shadow_X, shadow_Y,
                                                                                        test_size=0.5,
                                                                                        random_state=42)

        trainloader = DataLoader(trainset(*mix(target_train_X, target_train_Y, 1, epsilon)), batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(trainset(target_test_X, target_test_Y), batch_size=args.batch_size, shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = Model(4, 3)
        net.to(device)

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=args.decay)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        if not os.path.exists(args.save_to):
            os.makedirs(args.save_to)

        val_acc_max = 0
        for epoch in range(args.n_epochs):
            net.train()
            epoch_loss = 0
            acc = 0
            with tqdm(enumerate(trainloader, 0), total=len(trainloader)) as t:
                for i, data in t:
                    correct_items = 0

                    inputs, labels = data[0].to(device).long(), data[1].to(device).long()

                    optimizer.zero_grad()

                    outputs = net(inputs.float())
                    correct_items += torch.sum(torch.argmax(outputs, dim=-1) == labels).item()
                    acc_batch = correct_items / args.batch_size
                    acc += acc_batch

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    t.set_description("Epoch {:}/{:} Train".format(epoch + 1, 50))
                    t.set_postfix(accuracy="{:.3f}".format(acc / (i + 1)), loss="{:.3f}".format(epoch_loss / (i + 1)))

            net.eval()
            val_acc = 0
            with torch.no_grad():
                with tqdm(enumerate(testloader, 0), total=len(testloader)) as t:
                    for i, data in t:
                        correct_items = 0

                        inputs, labels = data[0].to(device), data[1].to(device)

                        outputs = net(inputs)
                        correct_items += torch.sum(torch.argmax(outputs, dim=-1) == labels).item()
                        val_acc_batch = correct_items / args.batch_size
                        val_acc += val_acc_batch

                        t.set_description("Epoch {:}/{:} VAL".format(epoch + 1, args.n_epochs))
                        t.set_postfix(accuracy="{:.3f}".format(val_acc / (i + 1)))

            torch.save(net.state_dict(), os.path.join(args.save_to, args.name + ".pth"))

        parser = argparse.ArgumentParser()
        parser.add_argument("--save_to", default='models', type=str)
        parser.add_argument("--name", default='iris', type=str)
        parser.add_argument("--shadow_num", default=1, type=int)
        parser.add_argument("--shadow_nepoch", default=10, type=int)
        parser.add_argument("--attack_nepoch", default=5, type=int)
        parser.add_argument("--topx", default=-1, type=int)
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

        attack_model = ConfidenceVector(shadow_models, args.attack_nepoch, device, args.topx)
        attack_model.train()
        b0 += attack_model.evaluate()
        c0 += attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))

    b.append(b0/n)
    c.append(c0/n)

df = pd.DataFrame({'epsilon': e, 'accuracy1': b, 'accuracy2': c})

df.to_csv("../data/iris_epsilon.csv", index=False, sep=',')
