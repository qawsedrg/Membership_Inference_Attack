import torch
import argparse
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import numpy as np
import nlpaug.augmenter.sentence as nas

from model import TextClassificationModel
from MIA.utils import trainset

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='agnews', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=15, type=int)

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


    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(int(_label))
            processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
            text_list.append(processed_text)
            # 用每个batch的前batchsize个元素作为分割点，长句被分割，点间段落求mean，embedd层输出为batchsize*embedsize
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list, offsets, label_list


    num_class = len(set([label for (label, text) in AG_NEWS(split='train')]))
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5)

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

    for p in [i / 10 for i in range(10)]:
        aug = naw.RandomWordAug(aug_p=p, aug_min=5, aug_max=10)
        with torch.no_grad():
            val_acc = 0
            with tqdm(enumerate(trainloader, 0), total=len(trainloader)) as t:
                for i, data in t:
                    correct_items = 0
                    auged = aug.augment(list(data[0]), num_thread=12)
                    data = collate_batch(list(zip(auged, data[1])))
                    label = data[-1]

                    outputs = model(*data[:-1])
                    correct_items += torch.sum(torch.argmax(outputs, dim=-1) == label).item()
                    val_acc_batch = correct_items / args.batch_size
                    val_acc += val_acc_batch

                    t.set_postfix(accuracy="{:.3f}".format(val_acc / (i + 1)))
            print(val_acc)
