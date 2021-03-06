import argparse
import os.path

import numpy as np
import torch
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from MIA.utils import trainset
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", default=50, type=int)
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='iris', type=str)
parser.add_argument('--decay', default=1e-2, type=float)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)
    Y = iris.target.astype(np.longlong)

    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)
    target_train_X, target_test_X, target_train_Y, target_test_Y = train_test_split(target_X, target_Y, test_size=0.5,
                                                                                    random_state=42)
    shadow_train_X, shadow_test_X, shadow_train_Y, shadow_test_Y = train_test_split(shadow_X, shadow_Y, test_size=0.5,
                                                                                    random_state=42)

    trainloader = DataLoader(trainset(target_train_X, target_train_Y), batch_size=args.batch_size,
                             shuffle=True)
    testloader = DataLoader(trainset(target_test_X, target_test_Y), batch_size=args.batch_size,
                            shuffle=False)

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

                inputs, labels = data[0].to(device), data[1].to(device).long()

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
'''
        if val_acc > val_acc_max:
            val_acc_max = val_acc
            torch.save(net.state_dict(), os.path.join(args.save_to, args.name + ".pth"))
'''
