import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

model_fname = "model.save"

MODEL_NAME = "clustering_base_gmm"


class ClusteringModel:
    def __init__(self, K, verbose=False, **kwargs) -> None:
        self.K = K
        self.verbose = verbose
        self.cluster_centers = None
        self.feature_names_in_ = None
        self.model = self.build_model()

    def build_model(self):
        model = GaussianMixture(
            n_components=self.K, verbose=self.verbose, random_state=0
        )
        return model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    def evaluate(self, x_test):
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    def save(self, model_path):
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    model = ClusteringModel.load(model_path)
    return model
