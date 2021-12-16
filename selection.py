# Different methods for selecting similar states

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import random


class KMeansSelector(KMeans):
    def __init__(
        self,
        config,
    ):
        super(KMeansSelector, self).__init__(
            n_clusters=config["n_clusters"],
            init=config["init"],
            n_init=config["n_init"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            verbose=config["verbose"],
            random_state=config["random_state"],
            copy_x=config["copy_x"],
            algorithm=config["algorithm"],
        )

    def fit(self, X, y=None):
        self._X = X.detach().cpu()
        self._data = self._X.numpy()
        # print(f'Length is {len(self._X)}')
        self._kmeans = super(KMeansSelector, self).fit(
            nn.functional.normalize(self._X, dim=1).numpy()
        )

    def get_similar_seqs(self, n_select, sample, seqs):
        assert n_select <= len(
            seqs
        ), "Number of sampled sequences should be >= number of sequences to be selected"
        _sample = nn.functional.normalize(sample, dim=1).detach().cpu().numpy()
        idx = self._kmeans.predict(_sample)
        _cluster = [seqs[i] for i in range(len(seqs)) if self._kmeans.labels_[i] == idx]

        try:
            similar = random.sample(_cluster, n_select)
        except:
            print("Selecting entire cluster")
            similar = _cluster

        return similar


_selector_dict = {"kmeans": KMeansSelector}


def get_selector(selector):
    return _selector_dict[selector]
