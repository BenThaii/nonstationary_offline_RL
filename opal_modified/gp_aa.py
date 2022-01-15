import torch
import torch.nn as nn
import os
import click
import d4rl, gym
import argparse
from utils import *
from nn_models import *
import numpy as np
from matplotlib import pyplot as plt
import torch.utils.data as data_utils
from matplotlib.pyplot import cm
from torch.utils.tensorboard import SummaryWriter
import time



'''this module trains the OPAL latent primitives'''


def eval_policy(env, model, num_eval, traj_length, tanh, is_cuda, target_goal=None):
	'''evaluating the policy by using the model to preduct and action at each step, and feed that to the environment, plot trajectory'''
	model.eval()
	
	if is_cuda and target_goal is not None:
		target_goal = target_goal.cuda()

	for _ in range(num_eval):				
		state = env.reset()
		state = torch.FloatTensor(state).unsqueeze(0)
		if is_cuda:
			state = state.cuda()		
		plt.imshow(env.render(mode='rgb_array'))

		done=False
		while not done:
			if target_goal is not None:
				latent = model.prior.act(latent=target_goal, state=state)
				if tanh:
					latent = torch.tanh(latent)
				latent = torch.cat([latent, target_goal], dim=1)
			else:				
				latent = model.prior.act(latent=None, state=state)
				if tanh:
					latent = torch.tanh(latent)													
			for t in range(traj_length):
				action = model.decoder.act(latent, state)
				action = action.cpu().numpy().flatten()
				state, _, done, _ = env.step(action)
				state = torch.FloatTensor(state).unsqueeze(0)
				if is_cuda:
					state = state.cuda()				
				plt.imshow(env.render(mode='rgb_array'))
				if done:
					break
	model.train()	

@click.command()
@click.option('--config', '-c', type=str, help='configuration_file')
def main(config):
	config = load_config(config)
	epoch_start_idx = 0
	continued_training = False
	# DEFAULT PARAMS
	if 'lr' not in config:
		config['lr'] = 1e-3
	if 'weight_decay' not in config:
		config['weight_decay'] = 0
	if 'latent_reg' not in config:
		config['latent_reg'] = 0
	if 'ar' not in config:
		config['ar'] = False					
	# END DEFAULT PARAMS

	torch.manual_seed(config['seed'])
	torch.cuda.manual_seed_all(config['seed'])
	np.random.seed(config['seed'])

	exp_name = config['exp_name']

	dir_name = 'gp_aa_experiments/{}'.format(config['env_name'])    
	save_dir, eval_dir, log_dir = clean_and_makedirs(dir_name=dir_name, exp_name=exp_name, seed=config['seed'])
	summary_writer = SummaryWriter(log_dir=log_dir)

	is_cuda = torch.cuda.is_available()
	# is_cuda = False
	is_render = config['render']

	env = gym.make(config['env_name'])
	if 'rgb_array' not in env.metadata['render.modes']:
		env.metadata['render.modes'].append('rgb_array')

	gp_aa_model = LMP(latent_dim=config['latent_dim'], state_dim=env.observation_space.shape[0], 
		action_dim=env.action_space.shape[0], hidden_dims=config['hidden_dims'], goal_idxs=config['goal_idxs'], tanh=config['tanh'], 
		latent_reg=config['latent_reg'], ar=config['ar'])

	if is_cuda:
		gp_aa_model = gp_aa_model.cuda()
	
	optimizer = torch.optim.Adam(gp_aa_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


	# # #load optimizer and network states for continued training
	# continued_training = True
	# print('continued')
	# saved_model_dir_path = "/home/ben/offline_RL/nonstationary_offline_RL/opal_modified/gp_aa_experiments/antmaze-medium-diverse-v0/lmp_tanh/20211225_1950-completed"
	# model_name = "lmp_tanh_100.pt"
	# epoch_start_idx = 101


	# saved_model_path = os.path.join(saved_model_dir_path, model_name)

	# checkpoint = torch.load(saved_model_path)
	# gp_aa_model.load_state_dict(checkpoint['gp_aa_model'])
	# optimizer.load_state_dict(checkpoint['opt'])

	# save_dir = saved_model_dir_path
	# eval_dir = saved_model_dir_path
	# log_dir = saved_model_dir_path

	# # #finished loading optimizer and network states
	




	state_traj, action_traj = get_traj_dataset(env, config['env_name'], config['traj_length'])
	state_traj, action_traj = torch.FloatTensor(state_traj), torch.FloatTensor(action_traj)

	# Clustering trajectories
	dataset = data_utils.TensorDataset(state_traj, action_traj)
	# loader = data_utils.DataLoader(dataset, config['batch_size'], shuffle=True)
	loader = data_utils.DataLoader(dataset, config['batch_size'])
	
	gp_aa_model.train()
	if continued_training == False and is_render:
		#view trajectory before training
		render_env = gym.wrappers.Monitor(env, os.path.join(eval_dir, 'no_train'), video_callable=lambda episode_id: is_render)
		eval_policy(render_env, gp_aa_model, config['num_eval'], config['traj_length'], config['tanh'], is_cuda) 	
	training_start = time.time()
	for epoch_num in range(epoch_start_idx, config['train_epochs']):
		epoch_start = time.time()
		total_loss = 0.
		total_kl_loss = 0.
		total_nll_loss = 0.
		total_abs_mean_dist = 0.
		total_encoder_mu_mean = 0.
		total_encoder_mu_std = 0.
		total_prior_mu_mean = 0.
		total_prior_mu_std = 0.
		for (idx, (state_b, action_b)) in enumerate(loader):
			if is_cuda:
				action_b = action_b.cuda()
				state_b = state_b.cuda()
			# kl_loss, nll_loss = gp_aa_model.calc_loss(state_b, action_b, is_cuda)
			kl_loss, nll_loss, abs_mean_dist, encoder_mu_mean, encoder_mu_std, prior_mu_mean, prior_mu_std = gp_aa_model.calc_loss(state_b, action_b, is_cuda)
			
			# Assumes config['reg'] >= 0
			if config['reg'] > 0:
				loss = nll_loss + config['reg']*kl_loss
			else:
				loss = nll_loss
			optimizer.zero_grad()	#from previous iteration
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			if config['reg'] > 0:
				total_kl_loss += kl_loss.item()
			total_nll_loss += nll_loss.item()
			total_abs_mean_dist += abs_mean_dist
			total_encoder_mu_mean += encoder_mu_mean
			total_encoder_mu_std += encoder_mu_std
			total_prior_mu_mean += prior_mu_mean
			total_prior_mu_std += prior_mu_std
		avg_loss = total_loss/(1+idx)
		avg_kl_loss = total_kl_loss/(1+idx)
		avg_nll_loss = total_nll_loss/(1+idx)
		avg_abs_mean_dist = total_abs_mean_dist/(1+idx)
		avg_encoder_mu_mean = total_encoder_mu_mean/(1+idx)
		avg_encoder_mu_std = total_encoder_mu_std/(1+idx)
		avg_prior_mu_mean = total_prior_mu_mean/(1+idx)
		avg_prior_mu_std = total_prior_mu_std/(1+idx)
		
		epoch_end = time.time()
		print(f"Avg loss for epoch {epoch_num} is {avg_loss}")
		print(f"Avg KL loss for epoch {epoch_num} is {avg_kl_loss}")
		print(f"Avg NLL loss for epoch {epoch_num} is {avg_nll_loss}")
		print(f"Avg abs distance of distribution mean for epoch {epoch_num} is {avg_abs_mean_dist}")
		print(f"Avg encoder mu mean {epoch_num} is {avg_encoder_mu_mean}")
		print(f"Avg encoder mu std {epoch_num} is {avg_encoder_mu_std}")
		print(f"Avg pior mu mean {epoch_num} is {avg_prior_mu_mean}")
		print(f"Avg prior mu std {epoch_num} is {avg_prior_mu_std}")
		
		print(f"Time epoch {epoch_num} is {epoch_end - epoch_start}")
		print(f"Total time elapsed is {epoch_end - training_start}\n")

		summary_writer.add_scalar("avg loss", avg_loss, epoch_num)
		summary_writer.add_scalar("avg KL loss", avg_kl_loss, epoch_num)
		summary_writer.add_scalar("avg NLL loss", avg_nll_loss, epoch_num)
		summary_writer.add_scalar("avg abs mean dist", avg_abs_mean_dist, epoch_num)
		summary_writer.add_scalar("avg encoder mu mean", avg_encoder_mu_mean, epoch_num)
		summary_writer.add_scalar("avg encoder mu std", avg_encoder_mu_std, epoch_num)
		summary_writer.add_scalar("avg prior mu mean", avg_prior_mu_mean, epoch_num)
		summary_writer.add_scalar("avg prior mu std", avg_prior_mu_std, epoch_num)

		if epoch_num % config['eval_interval'] == 0 and is_render:
			render_env = gym.wrappers.Monitor(env, os.path.join(eval_dir, f'epoch_num_{epoch_num}'), video_callable=lambda episode_id: is_render)
			eval_policy(render_env, gp_aa_model, config['num_eval'], config['traj_length'], config['tanh'], is_cuda) 

		save_dict = {'gp_aa_model':gp_aa_model.state_dict(), 'opt':optimizer.state_dict()}
		torch.save(save_dict, os.path.join(save_dir, f'{exp_name}_{epoch_num}.pt'))

	render_env = gym.wrappers.Monitor(env, os.path.join(eval_dir, f'epoch_num_{epoch_num}'), video_callable=lambda episode_id: is_render)
	eval_policy(render_env, gp_aa_model, config['num_eval'], config['traj_length'], config['tanh'], is_cuda) 		

if __name__ == '__main__':
	main()