# Different methods for selecting similar states

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from buffers import Transition

class KMeansSelector(KMeans):
    def __init__(
        self,
        config,
    ):
        super(KMeansSelector, self).__init__(
            n_clusters=config["n_clusters"],
            init=config['init'],
            n_init=config['n_init'],
            max_iter=config['max_iter'],
            tol=config['tol'],
            verbose=config['verbose'],
            random_state=config['random_state'],
            copy_x=config['copy_x'],
            algorithm=config['algorithm'],
        )

    def fit(self, X, y=None):
        self._X = X.detach().cpu()
        self._data = self._X.numpy()
        self._kmeans = super(KMeansSelector, self).fit(nn.functional.normalize(self._X, dim=1).numpy())

    def get_similar_states(self, n_select, sample, exp):
        _sample = nn.functional.normalize(sample, dim=1).detach().cpu().numpy()

        idx = self._kmeans.predict(_sample)
        _cluster = self._data[self._kmeans.labels_ == idx]
        similar = self._data[np.random.randint(_cluster.shape[0], size=n_select), :]
        
        transitions = [Transition(exp.obs[i], exp.action[i], exp.reward[i], exp.next_obs[i], exp.done[i], exp.reset[i], exp.imgn_code[i]) for i in similar]
        return transitions

_selector_dict = {
    'kmeans': KMeansSelector
}

def get_selector(selector):
    return _selector_dict[selector]