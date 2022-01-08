import argparse
import os.path

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model import TextClassificationModel

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='agnews', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_iter = AG_NEWS(split='train')
    tokenizer = get_tokenizer('basic_english')


    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)


    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1


    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            # 用每个batch的前batchsize个元素作为分割点，长句被分割，点间段落求mean，embedd层输出为batchsize*embedsize
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)


    train_iter = AG_NEWS(split='train')
    dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=collate_batch)

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)

    val_acc_max = 0
    for epoch in range(args.n_epochs):
        model.train()
        total_acc, total_count = 0, 0
        epoch_loss = 0
        acc = 0
        with tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)) as t:
            for i, (label, text, offsets) in t:
                correct_items = 0

                optimizer.zero_grad()

                outputs = model(text, offsets)

                correct_items += torch.sum(torch.argmax(outputs, dim=-1) == label).item()
                acc_batch = correct_items / args.batch_size
                acc += acc_batch

                loss = criterion(outputs, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                epoch_loss += loss.item()
                t.set_description("Epoch {:}/{:} Train".format(epoch + 1, args.n_epochs))
                t.set_postfix(accuracy="{:.3f}".format(acc / (i + 1)), loss="{:.3f}".format(epoch_loss / (i + 1)))

        model.eval()
        val_acc = 0
        with torch.no_grad():
            with tqdm(enumerate(valid_dataloader, 0), total=len(valid_dataloader)) as t:
                for i, (label, text, offsets) in t:
                    correct_items = 0

                    outputs = model(text, offsets)
                    correct_items += torch.sum(torch.argmax(outputs, dim=-1) == label).item()
                    val_acc_batch = correct_items / args.batch_size
                    val_acc += val_acc_batch

                    t.set_description("Epoch {:}/{:} VAL".format(epoch + 1, args.n_epochs))
                    t.set_postfix(accuracy="{:.3f}".format(val_acc / (i + 1)))
        if val_acc > val_acc_max:
            val_acc_max = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_to, args.name + ".pth"))
