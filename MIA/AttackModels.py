import os.path
import pickle
from typing import Optional
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

from MIA.ShadowModels import ShadowModels
from MIA.utils import trainset, train, attackmodel, forward, get_threshold


# todo 统一接口
class ConfidenceVector():
    def __init__(self, shadowmodel: ShadowModels, epoches: int, device: torch.device, topx: Optional[int] = -1,
                 transform: Optional = None):
        self.shadowdata = shadowmodel.data
        self.n_classes = int(max(torch.max(self.shadowdata.target_in).cpu().numpy(),
                                 torch.max(self.shadowdata.target_out).cpu().numpy()) + 1)
        self.topx = topx
        self.epoches = epoches
        self.device = device
        self.transform = transform

    def train(self, show=False):
        self.attack_models = []
        if self.topx == -1:
            for i in range(self.n_classes):
                train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                     self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
                train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                     torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(
                    self.device)
                attack_model = attackmodel(train_x.shape[-1])
                attack_model.to(self.device)
                optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
                loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
                print("\nTraining attack model for class {:}".format(i))
                attack_model = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                     epoches=self.epoches)
                self.attack_models.append(attack_model)
                if show:
                    fig = plt.figure()
                    train_x_in = self.shadowdata.data_in[self.shadowdata.target_in == i].cpu()
                    train_x_out = self.shadowdata.data_in[self.shadowdata.target_in == i].cpu()
                    X_in_tsne = TSNE(n_components=2).fit_transform(train_x_in)
                    X_out_tsne = TSNE(n_components=2).fit_transform(train_x_out)

                    ax = fig.add_subplot()

                    ax.scatter(X_out_tsne[:, 0], X_out_tsne[:, 1], marker='^', label="Not Trained")
                    ax.scatter(X_in_tsne[:, 0], X_in_tsne[:, 1], marker='o', label="Trained")

                    ax.set_xlabel('X_{:} Label'.format(i))
                    ax.set_ylabel('Y_{:} Label'.format(i))
                    ax.legend()

                    plt.show()
        else:
            train_x = torch.sort(torch.cat((self.shadowdata.data_in, self.shadowdata.data_out), dim=0), dim=-1)[0][:,
                      -self.topx:].to(self.device)
            train_y = torch.cat(
                (torch.ones(self.shadowdata.data_in.shape[0]), torch.zeros(self.shadowdata.data_out.shape[0]))).to(
                self.device)
            attack_model = attackmodel(train_x.shape[-1])
            attack_model.to(self.device)
            optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
            loader = DataLoader(trainset(train_x, train_y, None), batch_size=64, shuffle=True)
            attack_model = train(attack_model, loader, self.device, optimizer=optimizer, criterion=nn.BCELoss(),
                                 epoches=self.epoches)
            self.attack_models.append(attack_model)

    def __call__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        if self.topx == -1:
            classes = torch.max(X, dim=-1).indices
            result = torch.Tensor().to(self.device)
            data_x = torch.Tensor().to(self.device)
            data_y = torch.Tensor().to(self.device)
            for i in range(self.n_classes):
                x = X[classes == i]
                with torch.no_grad():
                    result = torch.cat((result, self.attack_models[i](x)))
                data_x = torch.cat((data_x, x), dim=0)
                if Y is not None:
                    y = Y[classes == i]
                    data_y = torch.cat((data_y, y), dim=0)
            return data_x, data_y, result
        else:
            with torch.no_grad():
                return X, Y, self.attack_models[0](X)

    '''
    def show(self):
        # todo
        data_in = torch.sort(self.shadowdata.data_in, dim=-1)[0][:, -self.topx:].cpu()
        data_out = torch.sort(self.shadowdata.data_out, dim=-1)[0][:, -self.topx:].cpu()
        attack_model = self.attack_models[0]

        ax = plt.axes(projection='3d')

        xp = np.linspace(0, 1, 100)
        yp = np.linspace(0, 1, 100)
        zp = np.linspace(0, 1, 100)
        x1, y1, z1 = np.meshgrid(xp, yp, zp)
        xyz = np.c_[x1.ravel(), y1.ravel(), z1.ravel()]
        with torch.no_grad():
            y_pred = attack_model(torch.from_numpy(xyz).to(self.device).float()).cpu().detach().numpy().reshape(
                x1.shape)
        x1, y1 = np.meshgrid(xp, yp)
        z = np.argmax((y_pred > 0.5).astype(int), axis=-1) / 100
        ax.contourf(x1, y1, z)

        ax.scatter3D(data_out[:, 0], data_out[:, 1], data_out[:, 2], marker='^', label="Not Trained")
        ax.scatter3D(data_in[:, 0], data_in[:, 1], data_in[:, 2], marker='o', label="Trained")

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()

        plt.show()
    '''
    def evaluate(self, target: Optional[nn.Module] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None):

        if target is not None:
            if self.topx == -1:
                target.to(self.device)
                loader = DataLoader(trainset(X_in, None, self.transform), batch_size=64, shuffle=False)
                output_in = forward(target, loader, self.device)
                loader = DataLoader(trainset(X_out, None, self.transform), batch_size=64, shuffle=False)
                output_out = forward(target, loader, self.device)
                _, _, result_in = self(F.softmax(output_in, dim=-1))
                _, _, result_out = self(F.softmax(output_out, dim=-1))
                correct = 0
                correct += torch.sum((result_in > 0.5)).cpu().numpy()
                correct += torch.sum((result_out < 0.5)).cpu().numpy()
                print(
                    "acc : {:.2f}".format(
                        correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
            else:
                target.to(self.device)
                loader = DataLoader(trainset(X_in, None, self.transform), batch_size=64, shuffle=False)
                output_in = forward(target, loader, self.device)
                output_in = torch.sort(output_in, dim=-1)[0][:, -self.topx:]
                loader = DataLoader(trainset(X_out, None, self.transform), batch_size=64, shuffle=False)
                output_out = forward(target, loader, self.device)
                output_out = torch.sort(output_out, dim=-1)[0][:, -self.topx:]
                _, _, result_in = self(F.softmax(output_in, dim=-1))
                _, _, result_out = self(F.softmax(output_out, dim=-1))
                correct = 0
                correct += torch.sum((result_in > 0.5)).cpu().numpy()
                correct += torch.sum((result_out < 0.5)).cpu().numpy()
                print(
                    "acc : {:.2f}".format(
                        correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
        else:
            if self.topx == -1:
                correct = 0
                for i in range(self.n_classes):
                    train_x = torch.cat((self.shadowdata.data_in[self.shadowdata.target_in == i],
                                         self.shadowdata.data_out[self.shadowdata.target_out == i]), dim=0)
                    train_y = torch.cat((torch.ones(torch.sum(self.shadowdata.target_in == i).cpu().numpy()),
                                         torch.zeros(torch.sum(self.shadowdata.target_out == i).cpu().numpy()))).to(
                        self.device)
                    attack_model = self.attack_models[i]
                    attack_model.to(self.device)
                    loader = DataLoader(trainset(train_x, train_y), batch_size=64, shuffle=False)
                    for data in loader:
                        with torch.no_grad():
                            correct += torch.sum((attack_model(data[0]) > 0.5).float() == data[1]).cpu().numpy()
                print(
                    "acc : {:.2f}".format(
                        correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))
            else:
                correct = 0
                train_x = torch.sort(torch.cat((self.shadowdata.data_in, self.shadowdata.data_out), dim=0), dim=-1)[0][
                          :, -self.topx:].to(self.device)
                train_y = torch.cat(
                    (torch.ones(self.shadowdata.data_in.shape[0]), torch.zeros(self.shadowdata.data_out.shape[0]))).to(
                    self.device)
                attack_model = self.attack_models[0]
                attack_model.to(self.device)
                loader = DataLoader(trainset(train_x, train_y), batch_size=64, shuffle=False)
                for data in loader:
                    with torch.no_grad():
                        correct += torch.sum((attack_model(data[0]) > 0.5).float() == data[1]).cpu().numpy()
                print(
                    "acc : {:.2f}".format(
                        correct / (self.shadowdata.data_in.shape[0] + self.shadowdata.data_out.shape[0])))


class BoundaryDistance():
    # todo test correcteness
    def __init__(self, shadowmodel: ShadowModels, device: torch.device, transform: Optional = None):
        self.shadowmodel = shadowmodel
        self.device = device
        self.acc_thresh = 0
        self.pre_thresh = 0
        self.transform = transform

    def train(self, show=False):
        if not os.path.exists("./dist_shadow_in") or not os.path.exists("./dist_shadow_out"):
            # todo many shadow models
            dist_shadow_in = self.train_base(self.shadowmodel.loader_train, self.shadowmodel[0],
                                             self.device)
            dist_shadow_out = self.train_base(self.shadowmodel.loader_test, self.shadowmodel[0],
                                              self.device)
            pickle.dump(dist_shadow_in, open("./dist_shadow_in", "wb"))
            pickle.dump(dist_shadow_out, open("./dist_shadow_out", "wb"))
        else:
            dist_shadow_in = pickle.load(open("./dist_shadow_in", "rb"))
            dist_shadow_out = pickle.load(open("./dist_shadow_out", "rb"))
        dist_shadow = np.concatenate((dist_shadow_in, dist_shadow_out))
        membership_shadow = np.concatenate((np.ones_like(dist_shadow_in), np.zeros_like(dist_shadow_out)))
        acc, self.acc_thresh, prec, self.pre_thresh = get_threshold(membership_shadow, dist_shadow)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))
        if show:
            plt.hist(dist_shadow_in, bins=100, range=[0, 2], label="in")
            plt.hist(dist_shadow_out, bins=100, range=[0, 2], label="out")
            plt.legend()
            plt.show()

    @staticmethod
    def train_base(loader, model, device):
        dist_adv = []
        model.to(device)
        with tqdm(enumerate(loader, 0), total=len(loader)) as t:
            for i, data in t:
                xbatch, ybatch = data[0].to(device), data[1].to(device)
                with torch.no_grad():
                    y_pred = F.softmax(model(xbatch), dim=-1)
                x_selected = xbatch[torch.argmax(y_pred, dim=-1) == ybatch, :]
                dist_adv.extend([0] * (xbatch.shape[0] - x_selected.shape[0]))
                # num_iteration
                x_adv_curr = hop_skip_jump_attack(model, x_selected, 2, num_iterations=1, verbose=0)
                # x_adv_curr = carlini_wagner_l2(model, x_selected,n_classes=100)
                d = torch.sqrt(torch.sum(torch.square(x_adv_curr - x_selected), dim=(1, 2, 3))).cpu().numpy()
                dist_adv.extend(d)
        return dist_adv

    def __call__(self, model, X: torch.Tensor):
        x_adv_curr = hop_skip_jump_attack(model, X, 2)
        d = torch.sqrt(torch.sum(torch.square(x_adv_curr - X), dim=(1, 2, 3))).cpu().numpy()
        return d > self.acc_thresh

    def evaluate(self, target: Optional[nn.Module] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None):
        if not os.path.exists("./dist_target_in") or not os.path.exists("./dist_target_out"):
            loader_in = DataLoader(trainset(X_in, Y_in, self.transform), batch_size=64, shuffle=False)
            loader_out = DataLoader(trainset(X_out, Y_out, self.transform), batch_size=64, shuffle=False)
            dist_target_in = self.train_base(loader_in, target, self.device)
            dist_target_out = self.train_base(loader_out, target, self.device)
            pickle.dump(dist_target_in, open("./dist_target_in", "wb"))
            pickle.dump(dist_target_out, open("./dist_target_out", "wb"))
        else:
            dist_target_in = pickle.load(open("./dist_target_in", "rb"))
            dist_target_out = pickle.load(open("./dist_target_out", "rb"))
        dist_target = np.concatenate((dist_target_in, dist_target_out))
        membership_target = np.concatenate((np.ones_like(dist_target_in), np.zeros_like(dist_target_out)))
        acc, _, _, _ = get_threshold(membership_target, dist_target, self.acc_thresh)
        _, _, prec, _ = get_threshold(membership_target, dist_target, self.pre_thresh)
        print("test_acc:{:},test_pre:{:}".format(acc, prec))


class Augmentation():
    def __init__(self, device: torch.device, trans: Optional = None, times: Optional = None,
                 transform: Optional = None):
        self.device = device
        # RandAugment ?
        self.trans = [T.RandomRotation(5), T.RandomAffine(degrees=0, translate=(0.1, 0.1))] if trans == None else trans
        self.times = [3 for _ in range(len(self.trans))] if times == None else times
        assert len(self.times) == len(self.trans)
        self.acc_thresh = 0
        self.pre_thresh = 0
        self.transform = transform

    def evaluate(self, target: Optional[nn.Module] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None,show=False):
        # 需要保证所有数据都用同一个变换吗，还是同一类型就行
        loader_train = DataLoader(trainset(X_in, Y_in, self.transform), batch_size=64, shuffle=False)
        loader_test = DataLoader(trainset(X_out, Y_out, self.transform), batch_size=64, shuffle=False)
        data_x_in = self.train_base(target, loader_train).cpu().numpy()
        data_x_out = self.train_base(target, loader_test).cpu().numpy()
        data_x = np.concatenate((data_x_in, data_x_out), axis=0)
        data_y = np.concatenate((np.ones(data_x_in.shape[0]), np.zeros(data_x_out.shape[0])))
        # 要改用监督学习吗，万一都是in，然后硬是分成两部分导致准确率低
        # 效果好吗？
        # 先tsne再分类？
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data_x)
        # kmeans=SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
        #                   assign_labels='kmeans').fit(data_x)
        acc = np.sum(kmeans.labels_ == data_y) / len(data_y)
        if acc < 0.5:
            data_y = 1 - data_y
        acc = np.sum(kmeans.labels_ == data_y) / len(data_y)
        prec = np.sum(
            [(kmeans.labels_ == 1)[i] and (kmeans.labels_ == data_y)[i] for i in range(len(data_y))]) / np.sum(
            kmeans.labels_)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))
        if show:
            fig = plt.figure()
            X_in_tsne = TSNE(n_components=2).fit_transform(data_x_in)
            X_out_tsne = TSNE(n_components=2).fit_transform(data_x_out)

            ax = fig.add_subplot()

            ax.scatter(X_out_tsne[:, 0], X_out_tsne[:, 1], c=kmeans.labels_[:len(data_x_in)], marker='^',
                       label="Not Trained")
            ax.scatter(X_in_tsne[:, 0], X_in_tsne[:, 1], c=kmeans.labels_[len(data_x_in):], marker='o', label="Trained")

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.legend()

            plt.show()

    def train_base(self, model, loader):
        # total*sum(times)
        result = torch.Tensor().to(self.device)
        for i, tran in enumerate(self.trans):
            for j in range(self.times[i]):
                torch.manual_seed(i * j)
                result_one_step = torch.Tensor().to(self.device)
                tran.to(self.device)
                model.to(self.device)
                with tqdm(loader, total=len(loader)) as t:
                    t.set_description("Transformation {:}|{:}".format(i, j))
                    for data in t:
                        xbatch, ybatch = tran(data[0].to(self.device)), data[1].to(self.device)
                        with torch.no_grad():
                            y_pred = F.softmax(model(xbatch), dim=-1)
                        result_one_step = torch.cat((result_one_step, torch.argmax(y_pred, dim=-1) == ybatch), dim=0)
                result = torch.cat((result, torch.unsqueeze(result_one_step, dim=-1)), dim=-1)
        return result

    def __call__(self, model, X: np.ndarray, Y: np.ndarray):
        transform = T.Compose(
            [T.ToTensor(),
             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        loader = DataLoader(trainset(X, Y, transform), batch_size=64, shuffle=False)
        out = self.train_base(model, loader).cpu().numpy()
        return KMeans(n_clusters=2, random_state=0).fit(out).labels_


class NoiseAttack():
    def __init__(self, shadowmodel: ShadowModels, device: torch.device, transform: Optional = None):
        self.shadowmodel = shadowmodel
        self.device = device
        self.acc_thresh = 0
        self.pre_thresh = 0
        self.stddev = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
        self.noisesamples = 50
        self.transform = transform

    def train(self, show=False):
        if not os.path.exists("./dist_shadow_in_noise") or not os.path.exists("./dist_shadow_out_noise"):
            # todo many shadow models
            dist_shadow_in = self.train_base(self.shadowmodel.loader_train, self.shadowmodel[0], self.stddev,
                                             self.noisesamples
                                             , self.device)
            dist_shadow_out = self.train_base(self.shadowmodel.loader_test, self.shadowmodel[0], self.stddev,
                                              self.noisesamples
                                              , self.device)
            pickle.dump(dist_shadow_in, open("./dist_shadow_in_noise", "wb"))
            pickle.dump(dist_shadow_out, open("./dist_shadow_out_noise", "wb"))
        else:
            dist_shadow_in = pickle.load(open("./dist_shadow_in_noise", "rb"))
            dist_shadow_out = pickle.load(open("./dist_shadow_out_noise", "rb"))
        dist_shadow = np.concatenate((dist_shadow_in, dist_shadow_out))
        membership_shadow = np.concatenate((np.ones_like(dist_shadow_in), np.zeros_like(dist_shadow_out)))
        acc, self.acc_thresh, prec, self.pre_thresh = get_threshold(membership_shadow, dist_shadow)
        print("train_acc:{:},train_pre:{:}".format(acc, prec))
        if show:
            plt.hist(dist_shadow_in, bins=100, range=[0, 1], label="in")
            plt.hist(dist_shadow_out, bins=100, range=[0, 1], label="out")
            plt.legend()
            plt.show()

    @staticmethod
    def train_base(loader, model, stddev, noise_samples, device):
        num_in = []
        model.to(device)
        with tqdm(enumerate(loader, 0), total=len(loader)) as t:
            for _, data in t:
                xbatch, ybatch = data[0].to(device), data[1].to(device)
                with torch.no_grad():
                    y_pred = F.softmax(model(xbatch), dim=-1)
                x_selected = xbatch[torch.argmax(y_pred, dim=-1) == ybatch, :]
                y_selected = ybatch[torch.argmax(y_pred, dim=-1) == ybatch]
                num_in.extend([0] * (xbatch.shape[0] - x_selected.shape[0]))
                # num_iteration
                for i in range(x_selected.shape[0]):
                    n = 0
                    for dev in stddev:
                        noise = torch.from_numpy(dev * np.random.randn(noise_samples, *x_selected.shape[1:])).to(device)
                        # 注意范围
                        x_noisy = torch.clamp(x_selected[i, :] + noise, -1, 1).float()
                        b_size = 100
                        with torch.no_grad():
                            for j in range(noise_samples // b_size + 1):
                                y_pred = F.softmax(model(x_noisy[j * b_size:(j + 1) * b_size]), dim=-1)
                                n += torch.sum(torch.argmax(y_pred, dim=-1) == y_selected[i]).item()
                    num_in.append(n / noise_samples)
        return num_in

    def evaluate(self, target: Optional[nn.Module] = None, X_in: Optional[np.ndarray] = None,
                 X_out: Optional[np.ndarray] = None,
                 Y_in: Optional[np.ndarray] = None,
                 Y_out: Optional[np.ndarray] = None):
        if not os.path.exists("./dist_target_in_noise") or not os.path.exists("./dist_target_out_noise"):
            loader_in = DataLoader(trainset(X_in, Y_in, self.transform), batch_size=64, shuffle=False)
            loader_out = DataLoader(trainset(X_out, Y_out, self.transform), batch_size=64, shuffle=False)
            dist_target_in = self.train_base(loader_in, target, self.stddev,
                                             self.noisesamples
                                             , self.device)
            dist_target_out = self.train_base(loader_out, target, self.stddev,
                                              self.noisesamples
                                              , self.device)
            pickle.dump(dist_target_in, open("./dist_target_in_noise", "wb"))
            pickle.dump(dist_target_out, open("./dist_target_out_noise", "wb"))
        else:
            dist_target_in = pickle.load(open("./dist_target_in_noise", "rb"))
            dist_target_out = pickle.load(open("./dist_target_out_noise", "rb"))
        dist_target = np.concatenate((dist_target_in, dist_target_out))
        membership_target = np.concatenate((np.ones_like(dist_target_in), np.zeros_like(dist_target_out)))
        acc, _, _, _ = get_threshold(membership_target, dist_target, self.acc_thresh)
        _, _, prec, _ = get_threshold(membership_target, dist_target, self.pre_thresh)
        print("test_acc:{:},test_pre:{:}".format(acc, prec))

    def __call__(self, model, X: torch.Tensor):
        y = torch.argmax(F.softmax(model(X), dim=-1), dim=-1)
        num_in = []
        for i in range(X.shape[0]):
            n = 0
            for dev in self.stddev:
                noise = torch.from_numpy(dev * np.random.randn(self.noisesamples, *X.shape[1:])).to(self.device)
                x_noisy = torch.clamp(X[i, :] + noise, -1, 1).float()
                b_size = 100
                with torch.no_grad():
                    for j in range(self.noisesamples // b_size + 1):
                        y_pred = F.softmax(model(x_noisy[j * b_size:(j + 1) * b_size]), dim=-1)
                        n += torch.sum(torch.argmax(y_pred, dim=-1) == y[i]).item()
            num_in.append(n / self.noisesamples)
        return np.array(num_in) > self.acc_thresh
