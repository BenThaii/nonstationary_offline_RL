import numpy as np
import os, shutil
import torch
import h5py
import torch.nn.functional as F

def extract_sub_demos(state_traj_lst, action_traj_lst, sub_length):
	"""Assumes sub_length < min length of all demos
	"""
	new_state_traj_lst = []
	new_action_traj_lst = []
	
	n_demos = len(state_traj_lst)
	avg_length = sum([len(traj) for traj in state_traj_lst])/n_demos
	n_sub_demos = int(n_demos*avg_length/sub_length)

	for _ in range(n_sub_demos):
		idx = np.random.randint(n_demos)
		len_demo = len(state_traj_lst[idx])
		start_idx = np.random.randint(len_demo - sub_length)
		new_state_traj_lst.append(state_traj_lst[idx][start_idx:start_idx+sub_length])
		new_action_traj_lst.append(action_traj_lst[idx][start_idx:start_idx+sub_length])
		
	return np.array(new_state_traj_lst), np.array(new_action_traj_lst)

def extract_sub_demos_sweep(state_traj_lst, action_traj_lst, sub_length):
	"""Assumes sub_length < min length of all demos
	"""
	new_state_traj_lst = []
	new_action_traj_lst = []
	
	n_demos = len(state_traj_lst)

	for idx in range(n_demos):
		len_demo = len(state_traj_lst[idx])
		for start_idx in range(len_demo - sub_length + 1):
			new_state_traj_lst.append(state_traj_lst[idx][start_idx:start_idx+sub_length])
			new_action_traj_lst.append(action_traj_lst[idx][start_idx:start_idx+sub_length])
		
	return np.array(new_state_traj_lst), np.array(new_action_traj_lst)

def clean_and_makedirs(dir_name, exp_name, seed, evaluate=False):
	main_dir = f'./{dir_name}/{exp_name}/seed_{seed}'
	save_path = os.path.join(main_dir, 'weights')
	eval_path = os.path.join(main_dir, 'eval')
	log_path = os.path.join(main_dir, 'log')
	
	if not evaluate: 
		if os.path.exists(main_dir):
			shutil.rmtree(main_dir)
		os.makedirs(save_path, exist_ok=True)
		os.makedirs(eval_path, exist_ok=True)
		os.makedirs(log_path, exist_ok=True)
	else:
		if os.path.exists(eval_path):
			shutil.rmtree(eval_path)
			os.makedirs(eval_path, exist_ok=True)

	return save_path, eval_path, log_path

def kld_gauss(mean_1, std_1, mean_2, std_2):
	kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
				(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
				std_2.pow(2) - 1)
	return	0.5 * torch.mean(torch.sum(kld_element, dim=1))

def load_config(path):
	with open(path, 'r', encoding='utf-8') as f:
		config = eval(f.read())
	return config

def get_pointmass_traj_dataset(env, traj_length, shuffle, image):
	dataset = env.env.get_dataset()
	states = dataset['observations']
	actions = dataset['actions']
	num_points = states.shape[0]
	assert num_points % traj_length == 0, "Invalid traj_length"
	states = states.reshape(-1, traj_length, states.shape[1])
	actions = actions.reshape(-1, traj_length, actions.shape[1])
	
	idxs = np.arange(states.shape[0])
	np.random.shuffle(idxs)	
	states = states[idxs]
	actions = actions[idxs]

	return states, actions

def get_images(env, large=False):
	main_path = '/home/aajay/.d4rl/datasets/'
	_, dataset_dir = os.path.split(env.env._dataset_url)
	dataset_dir = dataset_dir.split('.')[0]
	if large:
		img_dir = f'{dataset_dir}_img_large.hdf5'
	else:	
		img_dir = f'{dataset_dir}_img.hdf5' 
	img_path = os.path.join(main_path, img_dir)
	h5f = h5py.File(img_path, 'r')
	images = h5f['images'][:]
	return images	

def get_ant_traj_dataset(env, traj_length, image):
	dataset = env.env.get_dataset()
	states = dataset['observations'][:-1]
	actions = dataset['actions'][:-1]

	if image:
		images = get_images(env)
		images = images[:-1]
		images = images.reshape(-1, 1001, *images.shape[1:])		
		images = images[:,:-1]

	states = states.reshape(-1, 1001, states.shape[1])			
	actions = actions.reshape(-1, 1001, actions.shape[1])
	states = states[:,:-1]
	actions = actions[:,:-1]
	assert states.shape[1] % traj_length == 0, "Invalid traj_length"	
	
	if image:
		images = images.reshape(-1, traj_length, *images.shape[2:])	
	
	states = states.reshape(-1, traj_length, states.shape[2])
	actions = actions.reshape(-1, traj_length, actions.shape[2])	
	idxs = np.arange(states.shape[0])
	np.random.shuffle(idxs)	
	if image:
		images = images[idxs]
	
	states = states[idxs]
	actions = actions[idxs]
	
	if image:
		return (images, states[:,:,2:]), actions

	return states, actions

def get_ant_large_traj_dataset(env, traj_length, image):
	state_traj_lst = []
	action_traj_lst = []
	window = 1
	dataset = env.env.get_dataset()
	states = dataset['observations'][:-1]
	actions = dataset['actions'][:-1]
	states = states.reshape(-1, 1001, states.shape[1])
	actions = actions.reshape(-1, 1001, actions.shape[1])
	T = states.shape[1]

	for t in range(T-traj_length+1):
		state_traj_lst.append(states[:,t:t+traj_length])
		action_traj_lst.append(actions[:,t:t+traj_length])
	
	state_traj_lst = np.concatenate(state_traj_lst, axis=0)
	action_traj_lst = np.concatenate(action_traj_lst, axis=0)
	idxs = np.arange(state_traj_lst.shape[0])
	print(f"dataset size {state_traj_lst.shape[0]}")
	np.random.shuffle(idxs)
	state_traj_lst = state_traj_lst[idxs]
	action_traj_lst = action_traj_lst[idxs]
	return state_traj_lst, action_traj_lst 

def chop_traj(state_traj, action_traj, sub_length, window):
	state_traj_lst, action_traj_lst = [], []
	current_idx = 0
	traj_len = state_traj.shape[0]
	while current_idx < traj_len-sub_length:
		state_traj_lst.append(state_traj[current_idx:current_idx+sub_length][None])
		action_traj_lst.append(action_traj[current_idx:current_idx+sub_length][None])
		current_idx += window
	return state_traj_lst, action_traj_lst

def get_kitchen_traj_dataset(env, traj_length, image):
	state_traj_lst = []
	action_traj_lst = []
	window = 1
	dataset = env.env.get_dataset()
	states = dataset['observations']
	actions = dataset['actions']
	dones = dataset['terminals']
	done_idxs = np.where(dones == 1.0)[0]

	start_idx = 0
	for end_idx in done_idxs:
		state_traj = states[start_idx:end_idx]
		action_traj = actions[start_idx:end_idx]
		this_state_traj_lst, this_action_traj_lst = chop_traj(state_traj, action_traj, traj_length, window)
		state_traj_lst.extend(this_state_traj_lst)
		action_traj_lst.extend(this_action_traj_lst)
		start_idx = end_idx + 1

	state_traj_lst = np.concatenate(state_traj_lst, axis=0)
	action_traj_lst = np.concatenate(action_traj_lst, axis=0)
	idxs = np.arange(state_traj_lst.shape[0])
	np.random.shuffle(idxs)
	state_traj_lst = state_traj_lst[idxs]
	action_traj_lst = action_traj_lst[idxs]
	return state_traj_lst, action_traj_lst 

def get_pen_traj_dataset(env, traj_length, image):
	state_traj_lst = []
	action_traj_lst = []
	window = 1
	dataset = env.env.get_dataset()
	states = dataset['observations']
	actions = dataset['actions']
	
	actions = np.clip(actions, -0.98, 0.98)
	noise = np.random.uniform(low=-0.001, high=0.001, size=actions.shape)
	actions = actions + noise  	
	
	dones = dataset['timeouts']	
	done_idxs = np.where(dones == 1.0)[0]
	start_idx = 0
	for end_idx in done_idxs:
		state_traj = states[start_idx:end_idx]
		action_traj = actions[start_idx:end_idx]
		this_state_traj_lst, this_action_traj_lst = chop_traj(state_traj, action_traj, traj_length, window)
		state_traj_lst.extend(this_state_traj_lst)
		action_traj_lst.extend(this_action_traj_lst)
		start_idx = end_idx + 1

	state_traj_lst = np.concatenate(state_traj_lst, axis=0)
	action_traj_lst = np.concatenate(action_traj_lst, axis=0)
	idxs = np.arange(state_traj_lst.shape[0])
	np.random.shuffle(idxs)
	state_traj_lst = state_traj_lst[idxs]
	action_traj_lst = action_traj_lst[idxs]
	return state_traj_lst, action_traj_lst 

def get_hammer_traj_dataset(env, traj_length, image):
	state_traj_lst = []
	action_traj_lst = []
	window = 1
	dataset = env.env.get_dataset()
	states = dataset['observations']
	actions = dataset['actions']
	
	actions = np.clip(actions, -0.98, 0.98)
	noise = np.random.uniform(low=-0.001, high=0.001, size=actions.shape)
	actions = actions + noise

	dones = dataset['timeouts']	
	done_idxs = np.where(dones == 1.0)[0]
	start_idx = 0
	
	for end_idx in done_idxs:
		state_traj = states[start_idx:end_idx]
		action_traj = actions[start_idx:end_idx]
		this_state_traj_lst, this_action_traj_lst = chop_traj(state_traj, action_traj, traj_length, window)
		state_traj_lst.extend(this_state_traj_lst)
		action_traj_lst.extend(this_action_traj_lst)
		start_idx = end_idx + 1

	state_traj_lst = np.concatenate(state_traj_lst, axis=0)
	action_traj_lst = np.concatenate(action_traj_lst, axis=0)
	idxs = np.arange(state_traj_lst.shape[0])
	np.random.shuffle(idxs)
	state_traj_lst = state_traj_lst[idxs]
	action_traj_lst = action_traj_lst[idxs]
	return state_traj_lst, action_traj_lst 	

def get_carla_traj_dataset(env, traj_length, image):
	state_traj_lst = []
	action_traj_lst = []
	window = 1
	dataset = env.env.get_dataset()
	states = dataset['observations']
	actions = dataset['actions']

	dones = dataset['timeouts']	
	done_idxs = np.where(dones == 1.0)[0]
	start_idx = 0
	
	for end_idx in done_idxs:
		state_traj = states[start_idx:end_idx]
		action_traj = actions[start_idx:end_idx]
		this_state_traj_lst, this_action_traj_lst = chop_traj(state_traj, action_traj, traj_length, window)
		state_traj_lst.extend(this_state_traj_lst)
		action_traj_lst.extend(this_action_traj_lst)
		start_idx = end_idx + 1

	state_traj_lst = np.concatenate(state_traj_lst, axis=0)
	state_traj_lst = state_traj_lst.reshape(-1, traj_length, 48, 48, 3)
	action_traj_lst = np.concatenate(action_traj_lst, axis=0)
	idxs = np.arange(state_traj_lst.shape[0])
	np.random.shuffle(idxs)
	state_traj_lst = state_traj_lst[idxs]
	action_traj_lst = action_traj_lst[idxs]	
	return state_traj_lst, action_traj_lst 	

def get_traj_dataset(env, env_name, traj_length, image=False):
	if env_name in ('antmaze-umaze-diverse-v0', 'antmaze-large-diverse-v0', 'antmaze-medium-diverse-v0', 'antmaze-medium-diverse-v1', 'antmaze-medium-diverse-dense-v0', 'antmaze-medium-diverse-dense-v1'):
		return get_ant_traj_dataset(env, traj_length, image)
	elif env_name == 'maze2d-medium-v1':
		return get_pointmass_traj_dataset(env, traj_length, image)
	elif env_name in ('kitchen-mixed-v0', 'kitchen-partial-v0'):
		return get_kitchen_traj_dataset(env, traj_length, image)
	elif env_name in ('pen-cloned-v0', 'pen-human-v0'):
		return get_pen_traj_dataset(env, traj_length, image) 	
	elif env_name in ('hammer-cloned-v0'):
		return get_hammer_traj_dataset(env, traj_length, image) 
	elif env_name in ('carla-lane-v0', 'carla-town-v0'):
		return get_carla_traj_dataset(env, traj_length, image) 						
	else:
		raise NotImplementedError	


def spatial_softmax(features):
	"""Compute softmax over the spatial dimensions
	Compute the softmax over heights and width
	Args
	----
	features: tensor of shape [N, C, H, W]
	"""
	features_reshape = features.reshape(features.shape[:-2] + (-1,))
	output = F.softmax(features_reshape, dim=-1)
	output = output.reshape(features.shape)
	return output


def _maybe_convert_dict(value):
	if isinstance(value, dict):
		return ConfigDict(value)

	return value


class ConfigDict(dict):
	"""Configuration container class."""

	def __init__(self, initial_dictionary=None):
		"""Creates an instance of ConfigDict.
		Args:
			initial_dictionary: Optional dictionary or ConfigDict containing initial
			parameters.
		"""
		if initial_dictionary:
			for field, value in initial_dictionary.items():
				initial_dictionary[field] = _maybe_convert_dict(value)
		super(ConfigDict, self).__init__(initial_dictionary)

	def __setattr__(self, attribute, value):
		self[attribute] = _maybe_convert_dict(value)

	def __getattr__(self, attribute):
		try:
			return self[attribute]
		except KeyError as e:
			raise AttributeError(e)

	def __delattr__(self, attribute):
		try:
			del self[attribute]
		except KeyError as e:
			raise AttributeError(e)

	def __setitem__(self, key, value):
		super(ConfigDict, self).__setitem__(key, _maybe_convert_dict(value))


def get_random_color(pastel_factor=0.5):
	return [(x + pastel_factor) / (1.0 + pastel_factor)
			for x in [np.random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
	return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
	max_distance = None
	best_color = None
	for i in range(0, 100):
		color = get_random_color(pastel_factor=pastel_factor)
		if not existing_colors:
			return color
		best_distance = min([color_distance(color, c) for c in existing_colors])
		if not max_distance or best_distance > max_distance:
			max_distance = best_distance
			best_color = color
	return best_color


def get_n_colors(n, pastel_factor=0.9):
	colors = []
	for i in range(n):
		colors.append(generate_new_color(colors, pastel_factor=0.9))
	return colors