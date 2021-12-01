import torch
from sklearn.model_selection import train_test_split
import utils


class ShadowModels:
    def __init__(self, models, N, X, Y):
        self.models = models
        self.N = N
        self.X = X
        self.Y = Y
        self.shadowmodels = [self.train(model) for model in models]

    def train(self, model):
        pass
