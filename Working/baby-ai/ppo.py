from copy import deepcopy
from threading import Thread
import numpy

import torch
import torch.nn.functional as F

from babyai.rl.algos.base import BaseAlgo
from babyai.rl.utils import DictList, ParallelEnv

from buffers import ExperienceBuffer

hyper_params = {
    "in_channels": 3,
    "conv_channels": [32, 64, 64],
    "kernel_sizes": [8, 4, 3],
    "strides": [4, 2, 1],
    "paddings": [0, 0, 0],
    "use_maxpool": False,
    "head_sizes": None,
    "dropout": 0,
    "dyn_channels": 64,
    "pixels": 49,
    "hidden_size": 64,
    "limit": 1,
    "blocks": 0,
    "norm_type": "bn",
    "renormalize": 1,
    "residual": 0.0,
    "rollout_steps": 5,
    "trajs_per_fwd": 1,
    "traj_enc_size": 1024,
    "n_retrieval": 4,
    "d_k": 128,
    "enc_out_size": 128,
}

def rollout(forward_model, init_state, steps, num_trajs, dreams, i):
    init_state = init_state.permute(0, 3, 1, 2)
    out = forward_model.rollout(init_state, steps, num_trajs)

    dream = {
        'states': out[0],                      # [num_trajs_per_fwd_model, steps, batch_size, traj_enc_size]
        'rewards': out[1],
        'actions': out[2],
    }

    dreams[f'fwd_{i}'] = dream

class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None, forward_modules=None, selector=None, retrieval=False, n_retrieval=None, num_procs=None, aggregator=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info, device=device)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_retrieval = n_retrieval
        self.num_procs = num_procs

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

        if retrieval:
            self.forward_modules = forward_modules
            self.selector = selector
            self.replay_buffer = ExperienceBuffer(device=self.device, frames_per_process=self.num_frames_per_proc, num_processes=self.num_procs)
            self.aggregator = aggregator

        self.retrieval = retrieval
        self.flag = False

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                if self.retrieval and len(self.replay_buffer) < 32:
                    self.flag = True
                else:
                    self.flag = False

                state_emb, memory, extra_predictions = self.acmodel.get_state_embedding(preprocessed_obs, self.memory * self.mask.unsqueeze(1))

                if self.retrieval and not self.flag:
                    sample = self.replay_buffer.sample(self.n_retrieval)
                    sample_obs = DictList()
                    sample_obs.image = sample['obs']
                    sample_obs.instr = 0
                    r_state_emb, _, _ = self.acmodel.get_state_embedding(sample_obs, sample['memory'] * sample['mask'])

                    # Selecting similar observations from the replay buffer
                    sel_obs = self.selector(q=state_emb, k=r_state_emb, obs=sample['obs'])
                    # print(sel_obs.shape)

                    # Trajectory Rollouts
                    dreams = {}
                    rollout_threads = [Thread(target=rollout, args=(self.forward_modules[i], sel_obs[i], hyper_params['rollout_steps'], hyper_params['trajs_per_fwd'], dreams, i)) for i in range(len(self.forward_modules))]
                    for x in range(hyper_params['n_retrieval']): rollout_threads[x].start()
                    for y in range(hyper_params['n_retrieval']): rollout_threads[y].join()

                    del x
                    del y

                    # Aggregating the imagined states
                    fwd_states = [dreams[f'fwd_{i}']['states'] for i in range(hyper_params['n_retrieval'])]
                    fwd_states = [dream.reshape((-1,)+dream.shape[2:]) for dream in fwd_states]
                    # fwd_states = [tf_encoder(dream) for dream in fwd_states]
                    fwd_states = torch.cat(fwd_states, 0).permute(1, 0, 2)

                    # Selecting the relevant information from the imagined states
                    attn_info, _ = self.aggregator(q=state_emb, k=fwd_states)

                    # Residual Addition
                    state_emb = state_emb + attn_info

                # embedding, memory, extra_predictions = self.acmodel.get_state_embedding(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                model_results = self.acmodel(state_emb, memory, extra_predictions)
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = dist.sample()

            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            embedding, memory, extra_predictions = self.acmodel.get_state_embedding(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            next_value = self.acmodel(embedding, memory, extra_predictions)['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        if self.retrieval:
            self.replay_buffer.push(exps)

        return exps, log

    
    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss
                    state_emb, memory, extra_predictions = self.acmodel.get_state_embedding(sb.obs, memory * sb.mask)

                    if self.retrieval and not self.flag:
                        sample = self.replay_buffer.sample(self.n_retrieval)
                        sample_obs = DictList()
                        sample_obs.image = sample['obs']
                        sample_obs.instr = 0
                        r_state_emb, _, _ = self.acmodel.get_state_embedding(sample_obs, sample['memory'] * sample['mask'])

                        # Selecting similar observations from the replay buffer
                        sel_obs = self.selector(q=state_emb, k=r_state_emb, obs=sample['obs'])

                        # Trajectory Rollouts
                        dreams = {}
                        rollout_threads = [Thread(target=rollout, args=(self.forward_modules[i], sel_obs[i], hyper_params['rollout_steps'], hyper_params['trajs_per_fwd'], dreams, i)) for i in range(len(self.forward_modules))]
                        for x in range(hyper_params['n_retrieval']): rollout_threads[x].start()
                        for y in range(hyper_params['n_retrieval']): rollout_threads[y].join()

                        del x
                        del y

                        # Aggregating the imagined states
                        fwd_states = [dreams[f'fwd_{i}']['states'] for i in range(hyper_params['n_retrieval'])]
                        fwd_states = [dream.reshape((-1,) + dream.shape[2:]) for dream in fwd_states]
                        
                        # fwd_states = [tf_encoder(dream) for dream in fwd_states]
                        fwd_states = torch.cat(fwd_states, 0).permute(1, 0, 2)

                        # Selecting the relevant information from the imagined states
                        attn_info, _ = self.aggregator(q=state_emb, k=fwd_states)

                        # Residual Addition
                        state_emb = state_emb + attn_info

                    model_results = self.acmodel(state_emb, memory, extra_predictions)
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
