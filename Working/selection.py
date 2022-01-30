# Different methods for selecting similar states

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import random
from threading import Thread
from buffers import Sequence
from utils import Sparse_attention

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

    def sel_frm_cluster(self, seqs, idx, n_select, selected, i):
        _cluster = [seqs[i] for i in range(len(seqs)) if self._kmeans.labels_[i] == idx]
        similar = random.sample(_cluster, n_select)
        # print("Selecting entire cluster")
        # print(f"Cluster size is {len(_cluster)}")
        # print(f"Number of sequences to be selected is {n_select}")
        similar = _cluster

        selected[i] = similar

    def get_similar_seqs(self, n_select, sample, seqs, obsrvs_enc):
        assert n_select <= len(
            seqs
        ), "Number of sampled sequences should be >= number of sequences to be selected"
        assert len(self._X) == len(seqs)
        _sample = nn.functional.normalize(sample, dim=1).detach().cpu().numpy()
        idx = self._kmeans.predict(_sample)
        selected = {}
        selection_threads = [Thread(target=self.sel_frm_cluster, args=(seqs, idx[i], n_select, selected, i)) for i in range(len(idx))]

        for i in range(len(idx)): selection_threads[i].start()
        for i in range(len(idx)): selection_threads[i].join()

        zipped = zip(*[selected[k] for k in selected.keys()])
        similar = []

        for set in zipped:
            obs = np.stack([seq.obs for seq in set], axis=1)
            action = np.stack([seq.action for seq in set], axis=1)
            reward = np.stack([seq.reward for seq in set], axis=1)
            done = np.stack([seq.done for seq in set], axis=1)
            reset = np.stack([seq.reset for seq in set], axis=1)
            similar.append(Sequence(obs, action, reward, done, reset))

        return similar

class AttentionSelector(nn.Module):
    def __init__(self, config):
        super(AttentionSelector, self).__init__()
        self.topk = config["similar"]
        d_k = config["d_k"]
        d_model = config["enc_out_size"]
        self.temperature = np.power(d_k, 0.5)

        self.W_qs = nn.Linear(d_model, d_k)
        self.W_ks = nn.Linear(d_model, d_k)

        self.sa = Sparse_attention(top_k=self.topk)

    def forward(self, q, k, mask=None):
        q = self.W_qs(q)
        k = self.W_ks(k)

        attn = torch.mm(q, k.transpose(1, 0))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        sparse_attn = self.sa(attn)

        return sparse_attn

    def get_similar_seqs(self, n_select, sample, seqs, obsrvs_enc):
        assert n_select <= len(
            seqs
        ), "Number of sampled sequences should be >= number of sequences to be selected"
        assert len(seqs) == len(obsrvs_enc), "Length of seqs should be equal to length of obsrvs_enc"

        attn = self(q=sample, k=obsrvs_enc)

        batch_size = attn.shape[0]
        nz_indices = torch.nonzero(attn)[:, 1].reshape(batch_size, self.topk)
        
        similar = []

        for i in range(self.topk):
            indices = nz_indices[:, i]

            zipped = zip(*[seqs[i] for i in indices])
            field = []

            for set in zipped:
                field.append(np.stack(set, axis=1))
            
            similar.append(Sequence(*field))

        return similar

            
_selector_dict = {
                "kmeans": KMeansSelector,
                "attention": AttentionSelector,
                }


def get_selector(selector):
    return _selector_dict[selector]
