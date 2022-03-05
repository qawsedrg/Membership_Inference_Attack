import os
import torch
import argparse
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing
import nlpaug.augmenter.sentence as nas
import csv
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from MIA.utils import trainset

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='agnews_transformer', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=15, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--max_iter", default=10, type=int)
parser.add_argument("--num", default=multiprocessing.cpu_count() * 10, type=int)
parser.add_argument("--iter", default=10, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_class = len(set([label for (label, text) in AG_NEWS(split='train')]))
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,
                                                          num_labels=num_class).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))

    train_iter, test_iter = AG_NEWS()
    X = np.concatenate(([tup[1] for tup in list(train_iter)], [tup[1] for tup in list(test_iter)]))
    train_iter, test_iter = AG_NEWS()
    Y = np.concatenate(([tup[0] for tup in list(train_iter)], [tup[0] for tup in list(test_iter)])).astype(np.int64) - 1
    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)
    target_X_train, target_X_test, target_Y_train, target_Y_test = train_test_split(target_X, target_Y, test_size=0.5,
                                                                                    random_state=42)
    for _ in range(args.iter):
        acc_list = []


        def f(p):
            aug = naw.RandomWordAug(action='swap', aug_p=p, aug_min=0, aug_max=100)
            with torch.no_grad():
                val_acc = 0
                trainloader = DataLoader(trainset(target_X_train, target_Y_train), batch_size=args.batch_size,
                                         shuffle=True)
                with tqdm(enumerate(trainloader, 0), total=args.max_iter) as t:
                    for i, (text, label) in t:
                        correct_items = 0
                        auged = aug.augment(list(text), num_thread=12)
                        encoding = tokenizer(auged, return_tensors='pt', padding=True, truncation=True)
                        input_ids = encoding['input_ids'].to(device)
                        attention_mask = encoding['attention_mask'].to(device)
                        label = label.to(device)

                        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
                        correct_items += torch.sum(torch.argmax(outputs.logits, dim=-1) == label).item()
                        val_acc_batch = correct_items / args.batch_size
                        val_acc += val_acc_batch

                        t.set_postfix(accuracy="{:.3f}".format(val_acc / (i + 1)))
                        if i > args.max_iter:
                            return val_acc / (i + 1)


        numberOfThreads = multiprocessing.cpu_count()
        pool = ThreadPool(processes=numberOfThreads)
        l = [i / (args.num * 2) for i in range(args.num)]
        Chunks = np.array_split(l, len(l))
        results = pool.map_async(f, Chunks)
        pool.close()
        pool.join()
        for result in results.get():
            acc_list.append(result)

        with open("swap_tran", 'a') as f:
            writer = csv.writer(f)
            for i in range(len(l)):
                writer.writerow([l[i], acc_list[i]])
