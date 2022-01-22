import argparse
import os.path

import numpy as np
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split

from MIA.AttackModels import ConfidenceVector
from MIA.ShadowModels import ShadowModels
from model import TextClassificationModel

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='agnews', type=str)
parser.add_argument("--shadow_num", default=1, type=int)
parser.add_argument("--shadow_nepoch", default=5, type=int)
parser.add_argument("--attack_nepoch", default=5, type=int)
parser.add_argument("--topx", default=-1, type=int)

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
    target = TextClassificationModel(vocab_size, emsize, num_class)
    target.to(device)
    target.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))

    net = TextClassificationModel(vocab_size, emsize, num_class)
    net.to(device)

    train_iter, test_iter = AG_NEWS()
    X = np.concatenate(([tup[1] for tup in list(train_iter)], [tup[1] for tup in list(test_iter)]))
    train_iter, test_iter = AG_NEWS()
    Y = np.concatenate(([tup[0] for tup in list(train_iter)], [tup[0] for tup in list(test_iter)])).astype(np.int64) - 1
    target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)

    optimizer = torch.optim.SGD
    shadow_models = ShadowModels(net, args.shadow_num, shadow_X, shadow_Y, args.shadow_nepoch, device,
                                 collate_fn=collate_batch, opt=optimizer, lr=5)
    shadow_models.train()

    attack_model = ConfidenceVector(shadow_models, args.attack_nepoch, device, args.topx)
    attack_model.train()
    attack_model.evaluate()
    attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))
