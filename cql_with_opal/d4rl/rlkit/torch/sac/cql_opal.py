from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class CQLTrainer(TorchTrainer):
	def __init__(
			self,
			env,
			policy,
			qf1,
			qf2,
			target_qf1,
			target_qf2,
			latent_policy=None,					# vanilla CQL does not have this
			latent_policy_train_epochs=10000,	# vanilla CQL does not have this
			opal_unsupervised_policy = None,
			discount=0.99,
			reward_scale=1.0,

			policy_lr=1e-3,
			qf_lr=1e-3,
			optimizer_class=optim.Adam,

			soft_target_tau=1e-2,
			target_update_period=1,			# vanilla CQL does not have this
			plotter=None,
			render_eval_paths=False,

			use_automatic_entropy_tuning=True,
			latent_policy_train=False,		# vanilla CQL does not have this
			latent_policy_lr=3e-4,			# vanilla CQL does not have this
			target_entropy=None,
			alpha=1.0,						# vanilla CQL does not have this
			cql_start=0,					# vanilla CQL does not have this
			only_nll_before_start=False,	# vanilla CQL does not have this
			cql_temp=10.0,
			cql_alpha_lr=3e-4,				# vanilla CQL combines this with policy_lr
			cql_tau=10.0,
			cql_alpha=5.0,
			cql_alpha_min=0.02,
			cql_num_action_samples=10,
			use_cql_alpha_tuning=True,
			use_max_target=True,
			version=3             
	):
		'''
			input: 
				- policy: any RL policy
				- latent_policy: primitive policy
			optimization:
				- cql_alpha: balance the conservative penalty vs the bellman error
				- alpha: automatic enntropy tuning, maximum entropy objective, kinda like soft max (original paper:https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf, mentioned in: https://arxiv.org/pdf/1812.05905.pdf)
				- latent_policy: the (primitive) policy network
				- policy: policy network (ie: CQL), select action in feature space (z)
				- qf1: q-function 1
				- qf2: q-function 2

		'''
		super().__init__()
		self.env = env
		self.policy = policy												#CQL (or equivalent) agent - pick action in feature space z
		self.qf1 = qf1
		self.qf2 = qf2
		self.target_qf1 = target_qf1
		self.target_qf2 = target_qf2
		self.latent_policy = latent_policy									# primitive decoder from opal
		self.latent_policy_train = latent_policy_train
		self.latent_policy_train_epochs = latent_policy_train_epochs		# only train the primitive decoder from OPAL for this number of epochs

		self.opal_unsupervised_policy = opal_unsupervised_policy

		self.soft_target_tau = soft_target_tau
		self.target_update_period = target_update_period

		# START of CQL Params
		self._current_epoch = 0
		self.cql_start = cql_start
		self.only_nll_before_start = only_nll_before_start			##
		self.cql_temp = cql_temp
		self.cql_tau = cql_tau
		self.cql_alpha = cql_alpha						##
		self.cql_alpha_min = cql_alpha_min				##
		self.cql_num_action_samples = cql_num_action_samples				##
		self.use_cql_alpha_tuning = use_cql_alpha_tuning
		self.use_max_target = use_max_target				##

		if self.use_cql_alpha_tuning:
			# tune the \alpha to balance the conservative penalty vs the bellman backup
			self.log_cql_alpha = ptu.zeros(1, requires_grad=True)		#scalar value for alpha
			self.cql_alpha_optimizer = optimizer_class(
				[self.log_cql_alpha],
				lr=cql_alpha_lr,
			)

		self.version = version
		assert self.version in (2, 3)            
		# END of CQL Params

		# optimizer for maximum entropy RL to improve the performance in the worst case scenario
		self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
		if self.use_automatic_entropy_tuning:
			if target_entropy:
				self.target_entropy = target_entropy
			else:
				self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
			self.log_alpha = ptu.zeros(1, requires_grad=True)
			self.alpha_optimizer = optimizer_class(
				[self.log_alpha],
				lr=policy_lr,
			)

		#training the latent policy (primitive decoder from OPAL)
		if self.latent_policy_train:
			self.latent_policy_optimizer = optimizer_class(self.latent_policy.parameters(), lr=latent_policy_lr)

		self._alpha = alpha

		self.plotter = plotter
		self.render_eval_paths = render_eval_paths

		self.qf_criterion = nn.MSELoss()		# q-function
		self.vf_criterion = nn.MSELoss()		# value-function

		# policy network
		self.policy_optimizer = optimizer_class(
			self.policy.parameters(),
			lr=policy_lr,
		)

		# q-function network 1
		self.qf1_optimizer = optimizer_class(
			self.qf1.parameters(),
			lr=qf_lr,
		)

		# q-function network 2
		self.qf2_optimizer = optimizer_class(
			self.qf2.parameters(),
			lr=qf_lr,
		)

		self.discount = discount
		self.reward_scale = reward_scale
		self.eval_statistics = OrderedDict()
		self._n_train_steps_total = 0
		self._need_to_update_eval_statistics = True
		self.discrete = False

	def _get_tensor_values(self, obs, actions, network=None):
		action_shape = actions.shape[0]
		obs_shape = obs.shape[0]
		num_repeat = int (action_shape / obs_shape)
		obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
		preds = network(obs_temp, actions)
		preds = preds.view(obs.shape[0], num_repeat, 1)
		return preds

	def _get_policy_actions(self, obs, num_actions, network=None):
		obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
		new_obs_actions, _, _, new_obs_log_pi, *_ = network(
			obs_temp, reparameterize=True, return_log_prob=True,
		)
		return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)     

	def train_from_torch(self, batch):        
		rewards = batch['rewards']
		terminals = batch['terminals']
		obs = batch['observations']

		# this is the latent primitive encoding z generated by the OPAL primitive encoder
			# OPAL encoder (not offline policy) must be used to prepare the data in the "batch" that is used by the cql_trainer
			# This because we try to train the primitive policy to act like the behavior policy according to this latent z
		actions = batch['actions']	#also use actions (below), where action is now action feature (in z-space)				
		next_obs = batch['next_observations']	

		"""
		Policy and Alpha Loss
		"""
		# trains the latent (primitive decoder) policy 
			# this training phase enforces the primitive policy to act "like" the behavior policy in the trajectory, as characterized by latent z from the OPAL encoder 
			# specific to OPAL CQL
		if self.latent_policy_train and self._current_epoch < self.latent_policy_train_epochs:          
			obs_traj = batch['obs_traj']
			action_traj = batch['action_traj']
			T = obs_traj.shape[1]							# sequence length of the trajectory
			latents = actions.unsqueeze(1).repeat(1, T, 1)
			obs_traj = obs_traj.view(-1, obs_traj.shape[2])
			action_traj = action_traj.view(-1, action_traj.shape[2])
			latents = latents.view(-1, latents.shape[2])
			latent_log_prob = self.latent_policy.calc_log_prob(latents, obs_traj, action_traj)	# log prob of the primitive encoding
			latent_policy_loss = -latent_log_prob
			self.latent_policy_optimizer.zero_grad()
			latent_policy_loss.backward()
			self.latent_policy_optimizer.step()

		# the policy used for CQL is tanhGaussianPolicy
		new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
			obs, reparameterize=True, return_log_prob=True,
		)
		
		if self.use_automatic_entropy_tuning:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()			# because target entropy is defined with negative sign 
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
			alpha = self.log_alpha.exp()
		else:
			alpha_loss = 0
			alpha = self._alpha

		
		"""
		QF Loss
		"""
		q1_pred = self.qf1(obs, actions)
		q2_pred = self.qf2(obs, actions)
		
		# Make sure policy accounts for squashing functions like tanh correctly!
		new_next_actions, _, _, new_log_pi, *_ = self.policy(
			next_obs, reparameterize=True, return_log_prob=True,
		)
		new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
			obs, reparameterize=True, return_log_prob=True,
		)

		if not self.use_max_target:
			target_q_values = torch.min(
				self.target_qf1(next_obs, new_next_actions),
				self.target_qf2(next_obs, new_next_actions),
			)
		
		if self.use_max_target:
			"""when using max q backup"""
			next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=self.cql_num_action_samples, network=self.policy)
			target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
			target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
			target_q_values = torch.min(target_qf1_values, target_qf2_values)

		q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values        
		q_target = q_target.detach()
			
		qf1_loss = self.qf_criterion(q1_pred, q_target)
		qf2_loss = self.qf_criterion(q2_pred, q_target)

		random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.cql_num_action_samples, actions.shape[-1]).uniform_(-1, 1).to(ptu.device)
		curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.cql_num_action_samples, network=self.policy)
		new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.cql_num_action_samples, network=self.policy)
		q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
		q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
		q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
		q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
		q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
		q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

		cat_q1 = torch.cat(
					[q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
				)
		cat_q2 = torch.cat(
					[q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
				)
		std_q1 = torch.std(cat_q1, dim=1)
		std_q2 = torch.std(cat_q2, dim=1)

		if self.version == 3:
			# importance sampled version
			random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
				
			cat_q1 = torch.cat(
					[q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
				)
			cat_q2 = torch.cat(
					[q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
				)                
				
		"""log sum exp for the min"""
		min_qf1_loss = torch.logsumexp(cat_q1 / self.cql_temp, dim=1,).mean() * self.cql_alpha * self.cql_temp
		min_qf2_loss = torch.logsumexp(cat_q2 / self.cql_temp, dim=1,).mean() * self.cql_alpha * self.cql_temp
						
		"""Subtract the log likelihood of data"""
		min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.cql_alpha
		min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.cql_alpha

		if self.use_cql_alpha_tuning:
			cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=self.cql_alpha_min, max=1000000.0)
			min_qf1_loss = cql_alpha * (min_qf1_loss - self.cql_tau)
			min_qf2_loss = cql_alpha * (min_qf2_loss - self.cql_tau)
# 
			self.cql_alpha_optimizer.zero_grad()
			alpha_prime_loss = -0.5 * (min_qf1_loss + min_qf2_loss)
			alpha_prime_loss.backward(retain_graph=True)
			self.cql_alpha_optimizer.step()
				
		qf1_loss = qf1_loss + min_qf1_loss
		qf2_loss = qf2_loss + min_qf2_loss

		"""
		Update networks
		"""
		# Update the Q-functions iff 
		self.qf1_optimizer.zero_grad()
		qf1_loss.backward(retain_graph=True)
		self.qf1_optimizer.step()

		self.qf2_optimizer.zero_grad()
		qf2_loss.backward(retain_graph=False)
		self.qf2_optimizer.step()

		# update the policy network after having updated the q values
		q_new_actions = torch.min(self.qf1(obs, new_obs_actions), self.qf2(obs, new_obs_actions))

		policy_loss = (alpha*log_pi - q_new_actions).mean()			# want the policy to have high q values (minimize neg q) and max entropy -> low log pi

		if self._current_epoch < self.cql_start:
			"""Start with BC to the latent encoding z (for the CQL agent), not the physical actions themselves -> CQL tries to immitate the dataset during this period"""
			policy_log_prob = self.policy.log_prob(obs, actions)
			if self.only_nll_before_start:
				policy_loss = - policy_log_prob.mean()
			else:				
				policy_loss = (alpha*log_pi - policy_log_prob).mean()
		
		self._current_epoch += 1

		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		self.policy_optimizer.step()

		"""
		Soft Updates
		"""
		if self._n_train_steps_total % self.target_update_period == 0:
			ptu.soft_update_from_to(
				self.qf1, self.target_qf1, self.soft_target_tau
			)
			ptu.soft_update_from_to(
				self.qf2, self.target_qf2, self.soft_target_tau
			)

		"""
		Save some statistics for eval
		"""
		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			"""
			Eval should set this to None.
			This way, these statistics are only computed for one batch.
			"""
			self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
			self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
				policy_loss
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q1 Predictions',
				ptu.get_numpy(q1_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q2 Predictions',
				ptu.get_numpy(q2_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q Targets',
				ptu.get_numpy(q_target),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Log Pis',
				ptu.get_numpy(log_pi),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Policy mu',
				ptu.get_numpy(policy_mean),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Policy log std',
				ptu.get_numpy(policy_log_std),
			))
			if self.use_automatic_entropy_tuning:
				self.eval_statistics['Alpha'] = alpha.item()
				self.eval_statistics['Alpha Loss'] = alpha_loss.item()
			
			if self.use_cql_alpha_tuning:
				self.eval_statistics['cql_alpha'] = self.cql_alpha*cql_alpha.item()
				self.eval_statistics['cql_alpha_loss'] = alpha_prime_loss.item()
			else:
				self.eval_statistics['cql_alpha'] = self.cql_alpha

		self._n_train_steps_total += 1

	def get_diagnostics(self):
		return self.eval_statistics

	def end_epoch(self, epoch):
		# only update the evaluation statistics at the end of each epoch
		self._need_to_update_eval_statistics = True

	@property
	def networks(self):
		if self.latent_policy_train:
			# when we also train the latent policy, also return the latent network, otherwise, dont
			return [
				self.policy,
				self.qf1,
				self.qf2,
				self.target_qf1,
				self.target_qf2,
				self.latent_policy,            
			]
		else:
			return [
				self.policy,
				self.qf1,
				self.qf2,
				self.target_qf1,
				self.target_qf2,            
			]            

	def get_snapshot(self):
		if self.latent_policy_train:
			return dict(
				policy=self.policy,
				qf1=self.qf1,
				qf2=self.qf2,
				target_qf1=self.qf1,
				target_qf2=self.qf2,
				latent_policy=self.latent_policy,
				opal_unsupervised_policy = self.opal_unsupervised_policy
			)
		else:
			return dict(
				policy=self.policy,
				qf1=self.qf1,
				qf2=self.qf2,
				target_qf1=self.qf1,
				target_qf2=self.qf2,
			)              
