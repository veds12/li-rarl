import os
import random
import h5py
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch

Sequence = namedtuple("Sequence", ("obs", "action", "reward", "done", "reset"))
Transition = namedtuple(
    "Transition", ("obs", "action", "next_obs", "reward", "done")
)


class TransitionBuffer:
    def __init__(
        self,
        config,
    ):
        self._capacity = config["buffer_capacity"]
        self._memory = deque(maxlen=self._capacity)

    def push(self, state, action, next_state, reward, done):
        transition = Transition(state, action, next_state, reward, done)
        # print(f"Pushing these shapes: {transition.obs.shape, transition.action.shape, transition.next_obs.shape, transition.reward.shape, transition.done.shape, transition.imgn_code.shape}")
        self._memory.append(transition)

    def pop(self, end=None):
        if end == "left":
            return self._memory.popleft()
        elif end == "right" or end == None:
            return self._memory.pop()
        else:
            raise ValueError("end must be either left or right")

    def sample(self, batch_size):
        random_sample = random.sample(self._memory, batch_size)
        # grads = [code.imgn_code.grad_fn for code in random_sample]
        # codes = [code.imgn_code for code in random_sample]
        # print(grads, codes)
        return Transition(*[torch.cat(i) for i in [*zip(*random_sample)]])

    def save(self, filepath, name, VERBOSE=False):
        path = Path(filepath)
        path.mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, name)

        array_dict = [self._memory[i]._asdict() for i in range(len(self._memory))]
        for i in range(len(array_dict)):
            array_dict[i] = {k: v.cpu().numpy() for k, v in array_dict[i].items()}

        with h5py.File(str(path), 'w') as hf:
            for i in range(len(array_dict)):
                grp = hf.create_group(str(i))
                for key in array_dict[i].keys():
                    grp.create_dataset(key, data=array_dict[i][key])
        if VERBOSE:
            print(f'Transition buffer data stored at {path}')

    def load(self, filename):
        raise NotImplementedError

    def __len__(self):
        return len(self._memory)


class SequenceBuffer:
    # TODO:
    # - make more generic to allow incomplete episodes
    def __init__(
        self,
        config,
    ):
        self._capacity = config["buffer_capacity"]
        self._memory = deque(maxlen=self._capacity)

    def push(self, obs, action, reward, done, reset):
        transition = Sequence(obs, action, reward, done, reset)
        self._memory.append(transition)

    def pop(self, end=None):
        if end == "left":
            return self._memory.popleft()
        elif end == "right" or end == None:
            return self._memory.pop()
        else:
            raise ValueError("end must be either left or right")

    def sample_sequences(
        self, seq_len, n_sam_eps, reset_interval=0, skip_random=False,
    ):  # batch size = 1 for I2C style encoding
        sampled_episodes = random.sample(self._memory, n_sam_eps)
        sampled_seq = []

        for ep in sampled_episodes:
            data = {
                "image": ep.obs,
                "action": ep.action,
                "reward": ep.reward,
                "done": ep.done,
                "reset": ep.reset,
            }
            n = data["reward"].shape[0]
            data["reset"][0] = True
            data["reward"][0] = 0.0

            i = 0 if not skip_random else np.random.randint(n - seq_len + 1)

            if reset_interval:
                random_resets = self.randomize_resets(
                    data["reset"], reset_interval, seq_len
                )
            else:
                random_resets = np.zeros_like(data["reset"])

            while i < n:
                # print(f'i is {i}')
                # print(f'n is {n}')
                if i + seq_len > n:  # Pydreamer doesn't need this - why?
                    # print("Breaking")
                    break
                batch = {key: data[key][i : i + seq_len] for key in data.keys()}
                if np.any(random_resets[i : i + seq_len]):
                    assert not np.any(
                        batch["reset"]
                    ), "randomize_resets should not coincide with actual resets"
                    batch["reset"][0] = True

                i += seq_len
                # print(f'Len of image in this sequence: ', batch['image'].shape)
                sampled_seq.append(Sequence(*list(batch.values())))

        return sampled_seq

    def randomize_resets(self, resets, reset_interval, batch_length):
        assert resets[0]
        ep_boundaries = np.where(resets)[0].tolist() + [len(resets)]

        random_resets: np.ndarray = np.zeros_like(resets)  # type: ignore
        for i in range(len(ep_boundaries) - 1):
            ep_start = ep_boundaries[i]
            ep_end = ep_boundaries[i + 1]
            ep_steps = ep_end - ep_start

            # Cut episode into a random number of intervals

            max_intervals = (ep_steps // reset_interval) + 1
            n_intervals = np.random.randint(1, max_intervals + 1)
            i_boundaries = np.sort(
                np.random.choice(ep_steps - batch_length * n_intervals, n_intervals - 1)
            )
            i_boundaries = (
                ep_start + i_boundaries + np.arange(1, n_intervals) * batch_length
            )

            random_resets[i_boundaries] = True
            assert (resets | random_resets)[ep_start:ep_end].sum() == n_intervals

        return random_resets

    def save(self, filepath, name, VERBOSE=False):
        path = Path(filepath)
        path.mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, name)

        array_dict = [self._memory[i]._asdict() for i in range(len(self._memory))]

        with h5py.File(str(path), 'w') as hf:
            for i in range(len(array_dict)):
                grp = hf.create_group(str(i))
                for key in array_dict[i].keys():
                    grp.create_dataset(key, data=array_dict[i][key])
        if VERBOSE:
            print(f'Sequence buffer data stored at {path}')


    def load(self, filename):
        raise NotImplementedError

    def __len__(self):
        return len(self._memory)
