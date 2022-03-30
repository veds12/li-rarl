# Different methods for selecting similar states

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
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
        try:
            if config["attn_topk"] is not None:
                self.topk = config["attn_topk"]
            else:
                self.topk = config["retrieval_batch"]  
        except:
            self.topk = config["retrieval_batch"]
        
        d_k = config["d_k"]
        d_model = config["enc_out_size"]
        self.temperature = np.power(d_k, 0.5)
        self.n_actions = config['n_actions']

        self.W_qs = nn.Linear(d_model, d_k)        
        self.W_ks = nn.Linear(d_model, d_k)
        self.W_vs = nn.Linear(self.n_actions, self.n_actions)

        self.sa = Sparse_attention(top_k=self.topk)

    def forward(self, q, k, mask=None, **kwargs):
        # Shape of q = (batch_size, 2592)
        # Shape of k = (sample_size, 2592)
        k_init = k
        q = self.W_qs(q)                                            # Shape of q = (batch_size, d_k)
        k = self.W_ks(k[0])                                         # Shape of k = (sample_size, d_k)

        actions = F.one_hot(k_init[1].to(torch.int64), num_classes=self.n_actions).to(q.dtype)
        # value_net.eval()
        # values = value_net(k_init[0])                         
        v = self.W_vs(actions)                                             # Shape of v = (sample_size, d_model)

        attn = torch.mm(q, k.transpose(1, 0))
        attn = attn / self.temperature          # Shape of attn = (batch_size, sample_size)

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        sparse_attn = self.sa(attn)             # Shape of sparse_attn = (batch_size, sample_size)
        final_attn = torch.mm(sparse_attn, v)  # Shape of final_attn = (batch_size, d_model)

        return final_attn, sparse_attn

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

class ValueAttentionSelector(nn.Module):
    def __init__(self, config):
        super(ValueAttentionSelector, self).__init__()
        self.topk = config["val_topk"]
        config["n_retrieval"] = self.topk
        self.attn_selector = AttentionSelector(config)

    def forward(self, q, k, value_net, mask=None):
        # Shape of batch: (sample_size, 2592)
        
        with torch.no_grad():
            vals = value_net(k[0]).mean(dim=-1)
            ind = torch.topk(vals, self.topk)[1]
            selected_states = k[0][ind]
            selected_actions = k[1][ind]

        attn_state, sparse_attn = self.attn_selector(q=q, k=(selected_states, selected_actions))

        return attn_state, sparse_attn

class KNNSelector(nn.Module):
    def __init__(self, config):
        super(KNNSelector, self).__init__()
        self.knn = KNeighborsClassifier(n_neighbors=config['n_retrieval'])
        # self.W_vs = nn.Linear(config['n_actions'], config['n_actions'])
        self.n_actions = config['n_actions']
        self.topk = config['attn_topk']

    def forward(self, q, k, obs, **kwargs):
        dummy_labels = torch.tensor([random.randint(0, 1) for _ in range(k.shape[0])])
        self.knn.fit(k.cpu().detach(), dummy_labels)
        distances, indices = self.knn.kneighbors(q.cpu().detach())
        distances = torch.tensor(distances, dtype=q.dtype, device=q.device)
        # attn = nn.Softmax(dim=-1)(distances)

        # actions = F.one_hot(k[1].to(torch.int64), num_classes=self.n_actions).to(q.dtype)

        # v = actions[indices[0]]
        # v = self.W_vs(v)
        

        # attn_info = torch.mm(attn, v)

        selected_obs = obs[indices]

        if len(selected_obs.shape) != 5:
            selected_obs = selected_obs.unsqueeze(0)

        selected_obs = selected_obs.permute(1, 0, 2, 3, 4)
        return selected_obs

_selector_dict = {
                "kmeans": KMeansSelector,
                "attention": AttentionSelector,
                "value_attention": ValueAttentionSelector,
                "knn": KNNSelector,
                }


def get_selector(selector):
    return _selector_dict[selector]
