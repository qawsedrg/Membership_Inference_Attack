import multiprocessing
import os.path
import pickle
from multiprocessing.pool import ThreadPool
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from MIA.utils import trainset


class Augmentation():
    def __init__(self, device: torch.device, trans: List, times: List[int],
                 transform: Optional = None, collate_fn: Optional = None, batch_size: Optional[int] = 64):
        r"""
        Augmentation Attack model

        For each data, compute the a vector of shape sum(tran*time) 1 if augmented data is classified correctly, 0 if not

        Unsupervised classification of the vectors computed

        .. note::
            The attack performance is greatly affected by the choice of augmentation methods

        :param device: torch.device object
        :param trans: methods of augmentation to be performed
        :param times: number of classes
        :param transform: transformation to perform on images (before the augmentation)
        :param collate_fn: collate_fn used in DataLoader
        :param batch_size: batch_size used when inferring the augmented data
        """
        self.device = device
        self.trans = trans
        self.times = times
        assert len(self.times) == len(self.trans)
        self.acc_thresh = 0
        self.pre_thresh = 0
        self.transform = transform
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def evaluate(self, target: Optional[nn.Module] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None, show=False) -> Tuple[float, float]:
        # store the calculated vector, should be modified if needed
        # in - trained, out - not trained
        if not os.path.exists("./vec_x_in") or not os.path.exists("./vec_x_out"):
            loader_train = DataLoader(trainset(X_in, Y_in, self.transform), batch_size=self.batch_size, shuffle=False)
            loader_test = DataLoader(trainset(X_out, Y_out, self.transform), batch_size=self.batch_size, shuffle=False)
            vec_x_in = self.train_base(target, loader_train).cpu().numpy()
            vec_x_out = self.train_base(target, loader_test).cpu().numpy()
            pickle.dump(vec_x_in, open("./vec_x_in", "wb"))
            pickle.dump(vec_x_out, open("./vec_x_out", "wb"))
        else:
            vec_x_in = pickle.load(open("./vec_x_in", "rb"))
            vec_x_out = pickle.load(open("./vec_x_out", "rb"))
        vec_x = np.concatenate((vec_x_in, vec_x_out), axis=0)
        vec_y = np.concatenate((np.ones(vec_x_in.shape[0]), np.zeros(vec_x_out.shape[0])))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(vec_x)
        acc = np.sum(kmeans.labels_ == vec_y) / len(vec_y)
        # unsupervised, the label of output is undertimined
        if acc < 0.5:
            vec_y = 1 - vec_y
        acc = np.sum(kmeans.labels_ == vec_y) / len(vec_y)
        prec = np.sum((kmeans.labels_ == 1) * (kmeans.labels_ == vec_y)) / np.sum(kmeans.labels_ == 1)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))
        if show:
            fig = plt.figure()
            # TSNE visualizing of vectors, the random_state should be the same (difference of distribution is partially due to the choice to random_state)
            X_in_tsne = TSNE(n_components=2, random_state=0).fit_transform(vec_x_in)
            X_out_tsne = TSNE(n_components=2, random_state=0).fit_transform(vec_x_out)

            ax = fig.add_subplot()

            ax.scatter(X_out_tsne[:, 0], X_out_tsne[:, 1], marker='^',
                       label="Not Trained")
            ax.scatter(X_in_tsne[:, 0], X_in_tsne[:, 1], marker='o', label="Trained")

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.legend()

            plt.show()
        return (acc, prec)

    def train_base(self, model: nn.Module, loader: DataLoader) -> torch.Tensor:
        res = torch.Tensor().to(self.device)

        def f(Is):
            result = torch.Tensor().to(self.device)
            for i in Is:
                tran = self.trans[i]
                result_tran = torch.Tensor().to(self.device)
                for j in range(self.times[i]):
                    torch.manual_seed(i * j)
                    result_one_step = torch.Tensor().to(self.device)
                    tran.to(self.device)
                    model.to(self.device)
                    with tqdm(loader, total=len(loader)) as t:
                        # ith augmentation method | jth time
                        t.set_description("Transformation {:}|{:}".format(i + 1, j + 1))
                        for data in t:
                            if isinstance(data[0], torch.Tensor):
                                data = tran(data[0].to(self.device)), data[1].to(self.device)
                            else:
                                auged = tran(list(data[0]))
                                data = self.collate_fn(list(zip(auged, data[1])))
                            with torch.no_grad():
                                data = [d.to(self.device) for d in data]
                                xbatch = data[:-1]
                                ybatch = data[-1]
                                y_pred = F.softmax(model(*xbatch), dim=-1)
                            result_one_step = torch.cat((result_one_step, torch.argmax(y_pred, dim=-1) == ybatch),
                                                        dim=0)
                    result_tran = torch.cat((result_tran, torch.unsqueeze(result_one_step, dim=-1)), dim=-1)
                result = torch.cat((result, torch.sum(result_tran, dim=-1, keepdim=True) / self.times[i]), dim=-1)
            return result

        numberOfThreads = min(multiprocessing.cpu_count(), len(self.trans))
        pool = ThreadPool(processes=numberOfThreads)
        Chunks = np.array_split(range(len(self.trans)), numberOfThreads)
        results = pool.map_async(f, Chunks)
        pool.close()
        pool.join()
        for r in results.get():
            res = torch.cat((res, r), dim=-1)
        return res

    def __call__(self, model, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        loader = DataLoader(trainset(X, Y, self.transform), batch_size=self.batch_size, shuffle=False)
        out = self.train_base(model, loader).cpu().numpy()
        return KMeans(n_clusters=2, random_state=0).fit(out).labels_
