import argparse
import os.path

import numpy as np
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

from MIA.AttackModels import Augmentation
from model import TextClassificationModel
from MIA.utils import augmentation_wrapper

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

    aug1 = naw.WordEmbsAug(
        model_type='glove', model_path='D:\PSC\Membership_Inference_Attack\demo\glove.6B.50d.txt',
        action="substitute", aug_max=5, aug_min=2, aug_p=.5)  # 20min
    aug2 = naw.BackTranslationAug(device="cuda")  # 3h#out of memory
    aug3 = naw.ContextualWordEmbsAug(device="cuda")  # 1h30
    aug4 = naw.RandomWordAug()  # 5min

    trans = [augmentation_wrapper(aug4, 1)]

    attack_model = Augmentation(device, trans=trans, batch_size=64, collate_fn=collate_batch)
    attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42), show=True)
    membership = attack_model(target, target_X, target_Y)
