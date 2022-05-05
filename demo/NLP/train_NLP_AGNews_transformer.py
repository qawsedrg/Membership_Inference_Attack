import argparse
import os.path
import csv

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from tqdm import tqdm
from transformers import AdamW
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from MIA.utils import trainset

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", default=30, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='agnews_transformer', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # [CLS] 101
    # [SEP] 102
    # [PAD] 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    num_class = len(set([label for (label, text) in AG_NEWS(split='train')]))
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,
                                                          num_labels=num_class).to(device)
    for param in model.base_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_iter, test_iter = AG_NEWS()
    X = np.concatenate(([tup[1] for tup in list(train_iter)], [tup[1] for tup in list(test_iter)]))
    train_iter, test_iter = AG_NEWS()
    Y = np.concatenate(([tup[0] for tup in list(train_iter)], [tup[0] for tup in list(test_iter)])).astype(np.int64) - 1
    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)
    target_X_train, target_X_test, target_Y_train, target_Y_test = train_test_split(target_X, target_Y, test_size=0.5,
                                                                                    random_state=42)

    trainloader = DataLoader(trainset(target_X_train, target_Y_train), batch_size=args.batch_size,
                             shuffle=True)
    testloader = DataLoader(trainset(target_X_test, target_Y_test), batch_size=args.batch_size,
                            shuffle=False)

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)

    val_acc_max = 0
    for epoch in range(args.n_epochs):
        model.train()
        total_acc, total_count = 0, 0
        epoch_loss = 0
        acc = 0
        with tqdm(enumerate(trainloader, 0), total=len(trainloader)) as t:
            for i, (text, label) in t:
                correct_items = 0
                encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                label = label.to(device)

                optimizer.zero_grad()

                outputs = model(input_ids, attention_mask=attention_mask, labels=label)

                correct_items += torch.sum(torch.argmax(outputs.logits, dim=-1) == label).item()
                acc_batch = correct_items / args.batch_size
                acc += acc_batch

                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                epoch_loss += loss.item()
                t.set_description("Epoch {:}/{:} Train".format(epoch + 1, args.n_epochs))
                t.set_postfix(accuracy="{:.3f}".format(acc / (i + 1)), loss="{:.3f}".format(epoch_loss / (i + 1)))

        model.eval()
        val_acc = 0
        with torch.no_grad():
            with tqdm(enumerate(testloader, 0), total=len(testloader)) as t:
                for i, (text, label) in t:
                    correct_items = 0
                    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    label = label.to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=label)

                    correct_items += torch.sum(torch.argmax(outputs.logits, dim=-1) == label).item()
                    val_acc_batch = correct_items / args.batch_size
                    val_acc += val_acc_batch

                    t.set_description("Epoch {:}/{:} VAL".format(epoch + 1, args.n_epochs))
                    t.set_postfix(accuracy="{:.3f}".format(val_acc / (i + 1)))
        torch.save(model.state_dict(), os.path.join(args.save_to, args.name + ".pth"))
        with open("train_agnews_trans", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, acc / len(trainloader), val_acc / len(testloader)])
        if val_acc > val_acc_max:
            val_acc_max = val_acc
            # torch.save(model.state_dict(), os.path.join(args.save_to, args.name + ".pth"))
