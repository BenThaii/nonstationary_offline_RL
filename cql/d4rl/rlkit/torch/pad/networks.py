from torch import nn as nn
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy




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
    def __init__(self, obs_dim, action_dim, hidden_sizes, encoder= None, inv_network = None):
        encoder_output_size = encoder.output_size
        super().__init__(
            obs_dim = encoder_output_size,
            action_dim = action_dim,
            hidden_sizes = hidden_sizes
        )
        self.encoder = encoder
        self.inv_network = inv_network

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
