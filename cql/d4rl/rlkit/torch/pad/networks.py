import torch
from torch import nn as nn
import torch.nn.functional as F
from collections import deque, OrderedDict
import numpy as np
import copy
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import rollout, function_rollout



class EncodedQF(FlattenMlp):
    def __init__(self, action_size, output_size, hidden_sizes, encoder = None):
        encoder_output_size = encoder.output_size
        super().__init__(
            input_size= encoder_output_size + action_size, 
            output_size=output_size, 
            hidden_sizes=hidden_sizes,
        )
        self.encoder = encoder
    def forward(self, obs, actions, **kwargs):
        if self.encoder:
            encoded_obs = self.encoder(obs)
        else:
            encoded_obs = obs
        return super().forward(encoded_obs, actions, **kwargs)


class EncodedTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(self, 
            obs_dim, action_dim, hidden_sizes, 
            encoder= None, 
            inv_network = None, 
            encoder_lr = 1e-4,
            inv_lr = 1e-4,
        ):
        encoder_output_size = encoder.output_size
        super().__init__(
            obs_dim = encoder_output_size,
            action_dim = action_dim,
            hidden_sizes = hidden_sizes
        )
        self.encoder = encoder
        self.inv_network = inv_network

        # optimizers
        self.encoder_optimizer =  torch.optim.Adam(
            self.encoder.parameters(), lr=encoder_lr
        )
        self.inv_optimizer =  torch.optim.Adam(
            self.inv_network.parameters(), lr=inv_lr
        )

    def forward(self, obs, **kwargs):
        if self.encoder:
            encoded_obs = self.encoder(obs)
        else:
            encoded_obs = obs
        return super().forward(encoded_obs, **kwargs)

    def log_prob(self, obs, actions):
        if self.encoder:
            encoded_obs = self.encoder(obs)
        else:
            encoded_obs = obs
        return super().log_prob(encoded_obs, actions)
    
    def update_inv(self, obs, next_obs, actions):
        h = self.encoder(obs)
        h_next = self.encoder(next_obs)

        pred_actions = self.inv_network(h, h_next)
        inv_loss = F.mse_loss(pred_actions, actions)

        self.encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()
        inv_loss.backward()

        
        self.encoder_optimizer.step()
        self.inv_optimizer.step()

        # b = self.encoder.fc0.weight.data.clone()
        return inv_loss.item()
    




def rollout_pad(
        env,
        agent,
        use_pad,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        
        
        if use_pad:         # PAD
            o_torch = torch.FloatTensor(o[None]).to(ptu.device)
            next_o_torch = torch.FloatTensor(next_o[None]).to(ptu.device)
            a_torch = torch.FloatTensor(a[None]).to(ptu.device)
            agent.stochastic_policy.update_inv(o_torch, next_o_torch, a_torch)  

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )




def expl_rollout_pad(
        env,
        agent,
        qf,
        use_pad,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        # select the best action out of 10 actions
        with torch.no_grad():
            state = ptu.from_numpy(o.reshape(1, -1)).repeat(10, 1)
            action, _, _, _, _, _, _, _  = agent(state)
            q1 = qf(state, action)
            ind = q1.max(0)[1]
            a = ptu.get_numpy(action[ind]).flatten()

        next_o, r, d, env_info = env.step(a)

        if use_pad:         # PAD
            o_torch = torch.FloatTensor(o[None]).to(ptu.device)
            next_o_torch = torch.FloatTensor(next_o[None]).to(ptu.device)
            a_torch = torch.FloatTensor(a[None]).to(ptu.device)
            agent.update_inv(o_torch, next_o_torch, a_torch)

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,
    )




class MdpPathCollector_pad(PathCollector):      # used for policy evaluation
    def __init__(
            self,
            env,
            policy,
            use_pad,
            max_num_epoch_paths_saved=None,
            render=False,
            sparse_reward=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._sparse_reward = sparse_reward

        # PAD
        self.use_pad = use_pad

    def update_policy(self, new_policy):
        self._policy = new_policy
    
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            policy_fn=None,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            cur_policy = copy.deepcopy(self._policy)
            path = rollout_pad(
                self._env,
                cur_policy,
                use_pad = self.use_pad,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len

            ## Used to sparsify reward
            if self._sparse_reward:
                random_noise = np.random.normal(size=path['rewards'].shape)
                path['rewards'] = path['rewards'] + 1.0*random_noise 

            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )


class CustomMDPPathCollector_pad(PathCollector):        # used to assess policy exploration
    def __init__(
        self,
        env,
        policy,
        qf,
        use_pad,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._policy = policy
        self._qf = qf

        # PAD
        self.use_pad = use_pad

    def policy_fn(self, obs):
        """
        Used when sampling actions from the policy and doing max Q-learning
        """
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            state = ptu.from_numpy(obs.reshape(1, -1)).repeat(10, 1)
            action, _, _, _, _, _, _, _  = self._policy(state)
            q1 = self._qf(state, action)
            ind = q1.max(0)[1]
        return ptu.get_numpy(action[ind]).flatten()

    def collect_new_paths(
            self, policy_fn, max_path_length, 
            num_steps, discard_incomplete_paths
        ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            # create new copy of policy and qf, these will be modified during deployment
            cur_policy = copy.deepcopy(self._policy)
            cur_qf = copy.deepcopy(self._qf)
            cur_qf.encoder = cur_policy.encoder #share the encoder layer
            path = expl_rollout_pad(
                self._env,
                agent = cur_policy,
                qf = cur_qf,
                use_pad=self.use_pad,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
        )