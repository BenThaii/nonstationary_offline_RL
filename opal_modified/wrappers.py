import numpy as np
import itertools
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete

from collections import deque
from rlkit.torch.core import np_ify, torch_ify


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class HistoryEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, history_len):
        super().__init__(wrapped_env)
        self.history_len = history_len

        high = np.inf * np.ones(
            self.history_len * self.observation_space.low.size)
        low = -high
        self.observation_space = Box(low=low,
                                     high=high,
                                     )
        self.history = deque(maxlen=self.history_len)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history, reward, done, info

    def reset(self, **kwargs):
        state = super().reset()
        self.history = deque(maxlen=self.history_len)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history

    def _get_history(self):
        observations = list(self.history)

        obs_count = len(observations)
        for _ in range(self.history_len - obs_count):
            dummy = np.zeros(self._wrapped_env.observation_space.low.size)
            observations.append(dummy)
        return np.c_[observations]


class DiscretizeEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, num_bins):
        super().__init__(wrapped_env)
        low = self.wrapped_env.action_space.low
        high = self.wrapped_env.action_space.high
        action_ranges = [
            np.linspace(low[i], high[i], num_bins)
            for i in range(len(low))
        ]
        self.idx_to_continuous_action = [
            np.array(x) for x in itertools.product(*action_ranges)
        ]
        self.action_space = Discrete(len(self.idx_to_continuous_action))

    def step(self, action):
        continuous_action = self.idx_to_continuous_action[action]
        return super().step(continuous_action)

class LatentPolicyEnv(ProxyEnv, Env):
    '''Ben: probably use this to evaluate the policy, not to train, because it does not encode trajectory into z'''
    def __init__(self, wrapped_env, latent_policy, latent_action_space, traj_length, prior_policy=None):
        super().__init__(wrapped_env)
        self.latent_policy = latent_policy
        self.traj_length = traj_length
        self.prior_policy = prior_policy
        # self.discrete_latent = discrete_latent          #Ben: commented out because not used
        self.action_space = latent_action_space

    def step(self, latent, deterministic=False, get_traj=False):
        '''use the primitive decoder to actuate in the real environment for duration = traj_length'''
        if get_traj:
            obs_traj = []
            act_traj = []
        
        latent = torch_ify(latent[None])                    
        total_reward = 0.        
        obs = self.wrapped_env.env._get_obs()

        if self.prior_policy:
            reg_reward = self.prior_policy.calc_log_prob(latent=None, state=torch_ify(obs[None]), action=latent).item()
            total_reward += reg_reward                             
        
        for _ in range(self.traj_length):
            
            if get_traj:
                obs_traj.append(obs[None])
            
            obs = torch_ify(obs[None])            
            action = self.latent_policy.act(latent, obs, deterministic=deterministic)
            action = np_ify(action)[0]
            
            if get_traj:
                act_traj.append(action[None])            
            
            obs, reward, done, info = self.wrapped_env.step(action)            
            total_reward += reward

            if done:                
                break
        
        if get_traj:
            obs_traj = np.concatenate(obs_traj, axis=0)
            act_traj = np.concatenate(act_traj, axis=0)
            # return s_obs, total_reward, done, info, (obs_traj, act_traj)      #generate error
            return obs, total_reward, done, info, (obs_traj, act_traj)      

        # return s_obs, total_reward, done, info                                #generate error
        return obs, total_reward, done, info

    def step_state(self, state, latent, deterministic=True):
        # Only works when state is raw state and not image features
        self.reset_state(state)

        latent = torch_ify(latent[None])
        total_reward = 0.        
        obs = self.wrapped_env.env._get_obs()        
        
        for _ in range(self.traj_length):
            obs = torch_ify(obs[None])            
            action = self.latent_policy.act(latent, obs, deterministic=deterministic)
            action = np_ify(action)[0]
            obs, reward, done, info = self.wrapped_env.step(action)
            total_reward += reward             
            if done:                
                break

        return obs, total_reward, done, info     

    def reset_state(self, state):
        # Only works when state is raw state and not image features
        return self.wrapped_env.reset_state(state)

    def reset(self):
        obs = self.wrapped_env.reset()          
        return obs

    def _get_obs(self):
        obs = self.wrapped_env.env._get_obs()     
        return obs       

class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

