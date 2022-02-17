import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn
import nlpaug.augmenter.word as naw
import numpy as np
from tqdm import tqdm
import nlpaug.augmenter.sentence as nas
import nltk

train_iter = AG_NEWS(split='train')

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        # https://zhuanlan.zhihu.com/p/381461651
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


train_iter = AG_NEWS(split='train')
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
model.load_state_dict(torch.load("D:\PSC\Membership_Inference_Attack\demo\models\\agnews.pth"))

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

BATCH_SIZE = 64  # batch size for training
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True)


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text)).cuda()
        output = model(text, torch.tensor([0]).cuda())
        return output.argmax(1).item() + 1


aug = naw.RandomWordAug(aug_p=.5, aug_min=5, aug_max=10)
# aug = naw.ContextualWordEmbsAug(device="cuda")
for data in tqdm(valid_dataloader, total=len(valid_dataloader)):
    label, texts = data[0], data[1]
    for i, text in enumerate(texts):
        text_auged = aug.augment(text, n=10, num_thread=12)
        # print(nltk.translate.bleu_score.sentence_bleu(text_auged,text))
        labels = []
        for t in text_auged:
            labels.append(predict(t, text_pipeline))
        print(sum(np.array(labels) == label[i].item()))
