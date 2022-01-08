import numpy as np
import torch
from torch import nn
from multiprocessing.pool import ThreadPool
import multiprocessing


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


if __name__ == "__main__":
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.fc1 = nn.Linear(n, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 100)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    import argparse
    import os.path

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", default='D:\PSC\Membership_Inference_Attack\demo\models', type=str)
    parser.add_argument("--name", default='purchase', type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Model(600)
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(args.save_to, args.name + ".pth")))
    d = DataGenerator(net, 100, 600, 10, 10, "int", device)
    print(d())
