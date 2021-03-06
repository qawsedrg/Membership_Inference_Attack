import argparse
import csv
import os.path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchtext.datasets import IMDB
from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from MIA.Attack.ConfVector import ConfVector
from MIA.ShadowModels import ShadowModels

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='imdb_transformer', type=str)
parser.add_argument("--shadow_num", default=8, type=int)
parser.add_argument("--shadow_nepoch", default=30, type=int)
parser.add_argument("--attack_nepoch", default=5, type=int)
parser.add_argument("--topx", default=-1, type=int)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

num_class = len(set([label for (label, text) in IMDB(split='train')]))
target = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,
                                                       num_labels=num_class).to(device)
target.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))

net = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,
                                                    num_labels=num_class).to(device)

for param in net.base_model.parameters():
    param.requires_grad = False


def collate_batch(batch):
    texts, labels = list(zip(*batch))
    labels = torch.Tensor(labels)
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    return encoding['input_ids'], encoding['attention_mask'], labels.long()


train_iter, test_iter = IMDB()
X = np.concatenate(([tup[1] for tup in list(train_iter)], [tup[1] for tup in list(test_iter)]))
train_iter, test_iter = IMDB()
Y = np.concatenate(([0 if tup[0] == "neg" else 1 for tup in list(train_iter)],
                    [0 if tup[0] == "neg" else 1 for tup in list(test_iter)])).astype(np.int64)
target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)
target_X_train, target_X_test, target_Y_train, target_Y_test = train_test_split(target_X, target_Y, test_size=0.5,
                                                                                random_state=42)

optimizer = AdamW
shadow_models_pretrain = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, [5] * args.shadow_num, device,
                                      collate_fn=collate_batch, opt=optimizer, lr=1e-5, eval=False)
shadow_models_pretrain.train()

for n in range(1, args.shadow_num + 1):
    shadow_models = ShadowModels(shadow_models_pretrain, n, shadow_X, shadow_Y, args.shadow_nepoch, device,
                                 collate_fn=collate_batch, opt=optimizer, lr=1e-5)
    acc, val_acc = shadow_models.train()

    attack_model = ConfVector(shadow_models, args.attack_nepoch, device, args.topx)
    attack_model.train()
    shadow_acc, shadow_prec = attack_model.evaluate()
    target_acc, target_prec = attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5,
                                                                              random_state=42))

    with open("trans_imdb_conf_nshadowepoch", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([n * 5, np.average(acc), np.average(val_acc), shadow_acc, shadow_prec, target_acc, target_prec])
