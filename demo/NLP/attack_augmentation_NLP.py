import argparse
import os.path
import csv

import nlpaug.augmenter.word as naw
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

from MIA.Attack.Augmentation import Augmentation
from MIA.utils import augmentation_wrapper
from model import TextClassificationModel

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", default='models', type=str)
parser.add_argument("--name", default='imdb', type=str,
                    choices=['agnews', 'agnews_transformer', 'imdb', 'imdb_transformer'])
parser.add_argument("--method", default='delete', type=str, choices=["substitute", "swap", "delete"])
parser.add_argument("--num", default=10, type=int)
parser.add_argument("--repetition", default=1, type=int)

for n in range(1, 11):
    for i in range(1, 6):
        for method in ["substitute", "swap", "delete"]:
            args = parser.parse_args()
            args.method = method
            args.num = n
            args.repetition = i
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            train_iter = IMDB(split='train')
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


            num_class = len(set([label for (label, text) in IMDB(split='train')]))
            vocab_size = len(vocab)
            emsize = 64
            target = TextClassificationModel(vocab_size, emsize, num_class)
            target.to(device)
            target.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))

            net = TextClassificationModel(vocab_size, emsize, num_class)
            net.to(device)

            train_iter, test_iter = IMDB()
            X = np.concatenate(([tup[1] for tup in list(train_iter)], [tup[1] for tup in list(test_iter)]))
            train_iter, test_iter = IMDB()
            Y = np.concatenate(([0 if tup[0] == "neg" else 1 for tup in list(train_iter)],
                                [0 if tup[0] == "neg" else 1 for tup in list(test_iter)])).astype(np.int64)
            target_X, shadow_X, target_Y, shadow_Y = train_test_split(X, Y, test_size=0.5, random_state=42)
            target_X_train, target_X_test, target_Y_train, target_Y_test = train_test_split(target_X, target_Y,
                                                                                            test_size=0.5,
                                                                                            random_state=42)

            '''
            aug1 = naw.WordEmbsAug(
                model_type='glove', model_path='/glove.6B.50d.txt',
                action="substitute", aug_max=5, aug_min=2, aug_p=.5)  # 20min
            aug2 = naw.BackTranslationAug(device="cuda")  # 3h#out of memory
            aug3 = naw.ContextualWordEmbsAug(device="cuda")  # 1h30
            '''
            augs = [naw.RandomWordAug(action=args.method, aug_p=p, aug_min=0, aug_max=100) for p in
                    [i / (args.num * 2) for i in range(args.num)]]

            trans = [augmentation_wrapper(aug) for aug in augs]

            attack_model = Augmentation(device, trans=trans, batch_size=64, collate_fn=collate_batch,
                                        times=[args.repetition for _ in range(len(trans))])
            acc, prec = attack_model.evaluate(target,
                                              *train_test_split(target_X, target_Y, test_size=0.5, random_state=42),
                                              show=False, reuse=False)

            with open("RWA_imdb_rnn_method", 'a') as f:
                writer = csv.writer(f)
                writer.writerow([n, i, args.method, acc, prec])
