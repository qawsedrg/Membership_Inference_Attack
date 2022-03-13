import multiprocessing
from multiprocessing.pool import ThreadPool

import numpy as np
import torch
import torch.nn.functional as F


class DataGenerator:
    def __init__(self, target, nclass, n_features, k_max, n_records, dtype, device):
        self.target = target
        self.nclass = nclass
        self.n_features = n_features
        self.k_max = k_max
        self.n_records = n_records
        self.dtype = dtype
        self.device = device

    @staticmethod
    def features_generator(n_features: int, dtype: str, rang: tuple = (0, 1)):
        if dtype == "bool":
            x = np.random.randint(0, 2, n_features)
        if dtype == "int":
            x = np.random.randint(rang[0], rang[1] + 1, n_features)
        if dtype == "float":
            x = np.random.uniform(rang[0], rang[1], n_features)
        return x.reshape((1, -1))

    def generate(self, c):
        x = DataGenerator.features_generator(self.n_features, self.dtype)  # random record
        y_c_current = 0  # target model’s probability of fixed class
        n_rejects = 0  # consecutives rejections counter
        k = self.k_max
        k_min = 1
        max_iter = 1000
        conf_min = 0.9  # min prob cutoff to consider a record member of the class
        rej_max = 5  # max number of consecutive rejections
        with torch.no_grad():
            for _ in range(max_iter):
                y = np.squeeze(
                    F.softmax(self.target(torch.from_numpy(x).to(self.device).float()), dim=-1).cpu().numpy())
                y_c = y[c]
                if y_c >= y_c_current:
                    if (y_c > conf_min) and (c == np.argmax(y)):
                        return x
                    x_new = x
                    y_c_current = y_c
                    n_rejects = 0
                else:
                    n_rejects += 1
                    if n_rejects > rej_max:
                        k = max(k_min, int(np.ceil(k / 2)))
                        n_rejects = 0
                idx_to_change = np.random.randint(0, x.shape[1], size=k)
                new_feats = DataGenerator.features_generator(k, self.dtype, (0, 1))
                x_new[0, idx_to_change] = new_feats
                x = x_new
        return x

    def __call__(self):
        def f(Cs):
            result = []
            for c in Cs:
                result.append(self.generate(c))
            return np.array(result)

        x_syn = np.array([])
        numberOfThreads = multiprocessing.cpu_count()
        pool = ThreadPool(processes=numberOfThreads)
        Chunks = np.array_split(list(range(self.nclass)) * self.n_records, numberOfThreads)
        results = pool.map_async(f, Chunks)
        pool.close()
        pool.join()
        for result in results.get():
            x_syn = np.concatenate((x_syn, result)) if x_syn.shape != (0,) else result
        return x_syn
