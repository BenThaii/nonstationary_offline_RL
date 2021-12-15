from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            # onehot encoding
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            #Ben: parameterized action feature
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )


class EnvReplayBufferOpal(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            primitive_traj_length,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """

        
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        action_dim = get_dim(self._action_space)
        observation_dim = get_dim(self._ob_space)


        # new, specific to OPAL
        self._obs_traj = np.zeros((max_replay_buffer_size, primitive_traj_length, action_dim))
        self._action_traj = np.zeros((max_replay_buffer_size, primitive_traj_length, action_dim))
        #--

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=observation_dim,
            action_dim=action_dim,
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, obs_traj, action_traj, **kwargs):
        if isinstance(self._action_space, Discrete):
            # onehot encoding
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            #Ben: parameterized action feature
            new_action = action

        # new, specific to OPAL
        self._obs_traj[self._top] = obs_traj
        self._action_traj[self._top] = action_traj
        #--
        
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, obs_traj, action_traj, **kwargs):
        if isinstance(self._action_space, Discrete):
            # onehot encoding
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            #Ben: parameterized action feature
            new_action = action

        # new, specific to OPAL
        self._obs_traj[self._top] = obs_traj
        self._action_traj[self._top] = action_traj
        #--
        
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def random_batch(self, batch_size):
        # take a random batch from the buffer
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            obs_traj=self._obs_traj[indices],
            action_traj=self._action_traj[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch