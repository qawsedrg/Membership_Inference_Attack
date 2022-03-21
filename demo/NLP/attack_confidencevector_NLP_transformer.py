import argparse
import csv
import os.path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from MIA.Attack.ConfVector import ConfVector
from MIA.ShadowModels import ShadowModels

for n in range(5, 41, 5):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", default='models', type=str)
    parser.add_argument("--name", default='agnews_transformer', type=str)
    parser.add_argument("--shadow_num", default=1, type=int)
    parser.add_argument("--shadow_nepoch", default=n, type=int)
    parser.add_argument("--attack_nepoch", default=5, type=int)
    parser.add_argument("--topx", default=-1, type=int)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_iter = AG_NEWS(split='train')
    tokenizer = get_tokenizer('basic_english')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    num_class = len(set([label for (label, text) in AG_NEWS(split='train')]))
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


    train_iter, test_iter = AG_NEWS()
    X = np.concatenate(([tup[1] for tup in list(train_iter)], [tup[1] for tup in list(test_iter)]))
    train_iter, test_iter = AG_NEWS()
    Y = np.concatenate(([tup[0] for tup in list(train_iter)], [tup[0] for tup in list(test_iter)])).astype(np.int64) - 1
    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)
    target_X_train, target_X_test, target_Y_train, target_Y_test = train_test_split(target_X, target_Y, test_size=0.5,
                                                                                    random_state=42)

    optimizer = AdamW
    shadow_models = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, args.shadow_nepoch, device,
                                 collate_fn=collate_batch, opt=optimizer, lr=1e-5)
    acc, val_acc = shadow_models.train()

    attack_model = ConfVector(shadow_models, args.attack_nepoch, device, args.topx)
    attack_model.train()
    shadow_acc, shadow_prec = attack_model.evaluate()
    target_acc, target_prec = attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5,
                                                                              random_state=42))

    with open("trans_agnews_conf_nshadowepoch", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([n, acc, val_acc, shadow_acc, shadow_prec, target_acc, target_prec])
