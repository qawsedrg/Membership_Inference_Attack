import argparse
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    net = CIFAR(10)
    net.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(root='../data', train=True,
                                         download=True)
    test = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True)
    X, Y = np.concatenate((train.data, test.data)), np.concatenate((train.targets, test.targets)).astype(np.int64)
    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)
    target_X_train, target_X_test, target_Y_train, target_Y_test = train_test_split(target_X, target_Y, test_size=0.5,
                                                                                    random_state=42)
    trainloader = DataLoader(trainset(target_X_train, target_Y_train, transform), batch_size=args.batch_size,
                             shuffle=True, num_workers=1)
    testloader = DataLoader(trainset(target_X_test, target_Y_test, transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=1)


    criterion = nn.CrossEntropyLoss()
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

                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                correct_items += torch.sum(torch.argmax(outputs, dim=-1) == labels).item()
                acc_batch = correct_items / args.batch_size
                acc += acc_batch

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                t.set_description("Epoch {:}/{:} Train".format(epoch + 1, args.n_epochs))
                t.set_postfix(accuracy="{:.3f}".format(acc / (i + 1)), loss="{:.3f}".format(epoch_loss / (i + 1)))
        if (epoch + 1) % 1 == 0:
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
                print(epoch)
                val_acc_max = val_acc
                torch.save(net.state_dict(), os.path.join(args.save_to, args.name + ".pth"))
        '''
