from __future__ import annotations

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class DataGenerator:
    def __init__(self):
        pass

    @staticmethod
    def features_generator(n_features: int, dtype: str, rang: tuple = (0, 1)) -> np.ndarray:
        if dtype not in ("bool", "int", "float"):
            raise ValueError("Parameter `dtype` must be 'bool', 'int' or 'float'")

        if dtype == "bool":
            x = np.random.randint(0, 2, n_features)
        if dtype == "int":
            x = np.random.randint(rang[0], rang[1], n_features)
        if dtype == "float":
            x = np.random.uniform(rang[0], rang[1], n_features)
        return x.reshape((1, -1))

    @staticmethod
    def feature_randomizer(x: np.ndarray, k: int, dtype: str, rang: tuple) -> np.ndarray:
        idx_to_change = np.random.randint(0, x.shape[1], size=k)

        new_feats = DataGenerator.features_generator(k, dtype, rang)

        x[0, idx_to_change] = new_feats
        return x

    def synthesize(self,
                   target_model: nn.Module, c: int, k_max: int, dtype: str, n_features: int = None
                   ) -> np.ndarray | None:
        x = DataGenerator.features_generator(n_features, dtype=dtype)  # random record

        y_c_current = 0  # target model’s probability of fixed class
        n_rejects = 0  # consecutives rejections counter
        k = k_max
        k_min = 1
        max_iter = 1000
        conf_min = 0.8  # min prob cutoff to consider a record member of the class
        rej_max = 5  # max number of consecutive rejections
        with torch.no_grad():
            for _ in range(max_iter):
                ##
                y = target_model(x)  # query target model
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

                x = DataGenerator.feature_randomizer(x_new, k, dtype=dtype, rang=(0, 1))

        return None

    def synthesize_batch(target_model, fixed_cls, n_records, k_max, dtype):
        n_features = target_model.n_features_
        x_synth = np.zeros((n_records, n_features))

        for i in tqdm(range(n_records)):
            while True:  # repeat until synth finds record
                x_vec = DataGenerator.synthesize(target_model, fixed_cls, k_max, dtype)
                if isinstance(x_vec, np.ndarray):
                    break
            x_synth[i, :] = x_vec

        return x_synth
