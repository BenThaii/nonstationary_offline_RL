import torch
import torch.nn as nn
from rlkit.torch.opal.utils import kld_gauss
# from utils import kld_gauss

class FCNetwork(nn.Module):
	'''define a fully connected network to be used as a component of other Network Modules'''
	def __init__(self, inp_dim, hidden_dims, out_dim, act_fn=nn.ReLU()):
		'''
			input:
				- inp_dim: dimension of the input
				- hidden_dims: an array of dimensions of hidden layers
				- out_dim: output dimension
				- act_fcn: activation fcns to apply to apply after each hidden layer
					- note: does not apply to the output
		'''
		super(FCNetwork, self).__init__()
		self.inp_dim = inp_dim
		self.out_dim = out_dim
		self.hidden_dims = hidden_dims
		self.learn = True

		layer_lst = []
		in_dim = inp_dim

		for hidden_dim in hidden_dims:
			layer_lst.append(nn.Linear(in_dim, hidden_dim))
			layer_lst.append(act_fn)
			in_dim = hidden_dim

		layer_lst.append(nn.Linear(hidden_dim, out_dim))		#network output does not have activation function

		self.network = nn.Sequential(*layer_lst)

	def forward(self, inp):
	    return self.network(inp)

	@property
	def num_layers(self):    	
		return len(self.hidden_dims)+1

class PolicyNetwork(nn.Module):
	'''define the primitive policy network pi(a|s,z)'''
	def __init__(self, latent_dim, state_dim, hidden_dims, act_dim, masked_dim=0, act_fn=nn.ReLU()):
		'''
			input:
				- latent_dim: dimension of the latent vector z
				- state_dim: dimension of the state vector s
				- hidden_dims: array of hidden dimensions
				- act_dim: dimension of the action space
				- act_fn: activation after each hidden layer
				- masked_dim: number of channels at the beginning of the STATE vector that are masked (me: not sure why yet)
			output (of forward()): 
				- distribution for each action (array(mean), array(std))
			purpose:
				- create a fully connected network with 
					- input = (s,z), some component may be masked
					- output of dimension = 2 * action dimension:
						- first block: action mean
						- second block: action log standard deviation (log_std)
		'''
		super(PolicyNetwork, self).__init__()
		self.latent_dim = latent_dim
		self.state_dim = state_dim
		self.act_dim = act_dim
		self.masked_dim = masked_dim
		self.base = FCNetwork(inp_dim=latent_dim+state_dim-masked_dim, hidden_dims=hidden_dims, out_dim=2*act_dim, act_fn=act_fn)

	def forward(self, latent, state):
		'''
			input:
				- latent: row latent vector 
				- state: row state vector
			output: 
				- mean: action mean
				- std: action standard deviation
			assumption:
				- same set of actions can be taken at all states
			purpose:
				- for each (state, latent) tuple, return the mean and std for EACH ACTION the agent can take at that state	
		'''
		if state is None:
			inp = latent
		elif latent is None:
			inp = state[:, self.masked_dim:]
		else:
			inp = torch.cat([latent, state[:, self.masked_dim:]], dim=1)
		base_out = self.base(inp)
		mean = base_out[:,:self.act_dim]
		log_std = base_out[:,self.act_dim:]
		std = log_std.exp()
		return mean, std

	def act(self, latent, state, deterministic=False):
		'''
			input:
				- latent: row latent vector 
				- state: row state vector
				- deterministic: if deterministic: select the mean of the action, do not sample
			output:
				- the distribution over ALL ACTION that the agent will take -> the agent just need to select from this distribution
					- if deterministic: the probability of each action is determined by the mean of EACH action's distribution, 
					otherwise, sample from each action's distribution for that action's probabillity
		'''
		mean, std = self.forward(latent, state)
		if deterministic:
			return mean
		else:
			act_dist = torch.distributions.Normal(mean, std)			# create a n-distributions around n-means with n-standard deviations
			return act_dist.sample()

	def calc_log_prob(self, latent, state, action):
		'''
			input: 
				- latent: row latent vector 
				- action: onehot encoding of the action taken?
				- state: row state vector
			output:
				- log probability of that action encoding (how likely it is that that the selected action is selected, AND the unselected actions are not selected)
		'''
		mean, std =  self.forward(latent, state)
		act_dist = torch.distributions.Normal(mean, std)
		log_prob = act_dist.log_prob(action).sum(-1)					# calculate the log_prob of the actions (of both those selected, and those not selected). Sum of log_prob in log space is equivalent to product of prob in real space
		return log_prob.mean()

class ARPolicyNetwork(nn.Module):
	# Doesn't have masked dim
	def __init__(self, latent_dim, state_dim, act_dim, low_act=-1.0, up_act=1.0, act_fn=nn.ReLU()):
		'''
		input: 
			- latent_dim: dimension of latent vector 
			- state_dim: dimension of state vector
			- act_dim: dimension of the action (onehot) vector
		purpose:
			- TODO: ask Dr.Lee: why do they use such a convoluted method?
			- predict the one-hot encoding of the next action through a convoluted method where each dimension of the action encoding is selected based on the decoded representation of the previous action dimensions
		'''
		super(ARPolicyNetwork, self).__init__()
		self.latent_dim = latent_dim
		self.state_dim = state_dim
		self.act_dim = act_dim
		
		self.state_embed_hid = 256			# number of hidden units in the 4 hidden layers to embed the state
		self.action_embed_hid = 128			# number of hidden units in the action encoding hidden layer -> to predict each action encoding dimension
		self.out_lin = 128					# number of dimensions of the representation of the previous action dimensions, as decoded by lin_mod
		self.num_bins = 80					# number of bins of each action dimension

		self.up_act = up_act
		self.low_act = low_act
		self.bin_size = (self.up_act-self.low_act)/self.num_bins
		self.ce_loss = nn.CrossEntropyLoss()			# used to find the likelihood of each action. Cross Entropy is already a log loss

		# Component 1: (s+z) -> 4 Fully Connected Network with hidden dimension = state_embed_hid
			# putpose: generate a state embedding
		self.state_embed = nn.Sequential(
							nn.Linear(self.state_dim+self.latent_dim, self.state_embed_hid),
							act_fn,
							nn.Linear(self.state_embed_hid, self.state_embed_hid),
							act_fn,
							nn.Linear(self.state_embed_hid, self.state_embed_hid),
							act_fn,
							nn.Linear(self.state_embed_hid, self.state_embed_hid),														
						)

		# list 2: list of unconnected components with input dim varying from 1 to act_dim, and output dim = out_lin
			# this is because each time we select a set of action encoding dimensions (from dimension 0 to self.act_dim), we pass through the components defined in this list to get out_lin
			# purpose: decode previous action dimensions into a tensor of dimension = out_lin, which is concatenated to the state_embeeding to obtain the next action dimension
		self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.act_dim)])
		
		# list 3: list of SINGLE Sequential component that converts from state_embedding -> action_embedding -> value for each bin of actions for each dimension
			# structure: (state_embed_hid -> action_embed_hid) -> activation fcn -> (action_embed_hid -> num_bins) 
			# purpose: generate a distribution over all bins for each action dimension -> allow us to select a value for each action dimension
		self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(self.state_embed_hid, self.action_embed_hid), act_fn, nn.Linear(self.action_embed_hid, self.num_bins))])		

		# add components to list 3, each component has structure: (state_embed_hid + out_lin -> action_embed_hid) -> activation fcn -> (action_embed_hid -> num_bins)  (2 layer neural net)
			# this is because for the action dimensions > 0, we can use previous actions dimensions to create a representation of dimension = out_lin
		for _ in range(1, self.act_dim):
			self.act_mod.append(nn.Sequential(nn.Linear(self.state_embed_hid + self.out_lin, self.action_embed_hid), act_fn, nn.Linear(self.action_embed_hid, self.num_bins)))

	def forward(self, latent, state, deterministic=False):
		'''
			input:
				- latent: row latent vector 
				- state: row state vector
				- deterministic: TRUE -> select the middle value of the bin; FALSE -> sample across all values in the bin according to a uniform distribution
			output:
				- an action one-hot encoding
		'''

		if state is None:
			state_inp = latent
		elif latent is None:
			state_inp = state
		else:
			state_inp = torch.cat([latent, state], dim=1)					# note: no more masking

		# aim: select the first action dimension a_0
		state_d = self.state_embed(state_inp)								# create a state embedding using component 1
		lp_0 = self.act_mod[0](state_d)										# create logits for each bin by passing state embedding through the first component of list 3 to create a tensor lp_0 of dimension = num_bins
		l_0 = torch.distributions.Categorical(logits=lp_0).sample()			# sample from the distribution described in lp_0 to get the left edge of the bin
		
		if deterministic:
			a_0 = self.low_act + (l_0+0.5)*self.bin_size					# if deterministic: select the middle of the bin
		else:
			a_0 = torch.distributions.Uniform(self.low_act + l_0*self.bin_size, self.low_act + (l_0+1)*self.bin_size).sample()		#if not deterministic: select a_0 value from the uniform distribution over the bin's domain (left -> right edge)

		a = [a_0.unsqueeze(1)]												# obtain the selected action

		#aim: select the subsequent action dimensions of the action embedding
		for i in range(1, self.act_dim):
			lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i-1](torch.cat(a, dim=1))], dim=1))		# each action dimension is selected with input = state + representation of the previous action dimension as decoded by self.lin_mod network
			l_i = torch.distributions.Categorical(logits=lp_i).sample()		

			if deterministic:																									# same sampling process as above for a_0
				a_i = self.low_act + (l_i+0.5)*self.bin_size
			else:
				a_i = torch.distributions.Uniform(self.low_act + l_i*self.bin_size, self.low_act + (l_i+1)*self.bin_size).sample()			

			a.append(a_i.unsqueeze(1))
		
		return torch.cat(a, dim=1) 

	def act(self, latent, state, deterministic=False):
		'''
			input:
				- latent: row latent vector 
				- state: row state vector
				- deterministic: TRUE -> select the middle value of the bin; FALSE -> sample across all values in the bin according to a uniform distribution
			output:
				- action embedding
			Note: unlike the above where the mean and std is returned and we had to sample, the return of the forward() function is already an action embedding -> dont have to sample
		'''
		return self.forward(latent, state, deterministic)

	def calc_log_prob(self, latent, state, action):
		'''
			input:
				- latent: row latent vector 
				- state: row state vector
				- action: an action embedding
			output:
				- returns how likely the action embedding is generated, given the state and latent vector
			process:
				- use act_mod to predict the likelihood of each bin of action, then use ce_loss to determine how likely the action of interest is selected from that distribution over all bins of actions

		'''
		l_action = ((action - self.low_act)//self.bin_size).long()			#convert the action to its corresponding bin (in each dimension). This is because act_mod provides the discretized logits for each bin, for within each bin

		if state is None:
			state_inp = latent
		elif latent is None:
			state_inp = state
		else:
			state_inp = torch.cat([latent, state], dim=1)

		state_d = self.state_embed(state_inp)						#obtain the state embedding
		
		#for log_prob, use negative of ce_loss because of the definition of ce_loss: more likely action -> less loss -> lower ce_loss 
		log_prob = -self.ce_loss(self.act_mod[0](state_d), l_action[:,0])		# get the ce_loss of the 0th action
		for i in range(1, self.act_dim):
			log_prob -= self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i-1](action[:,:i])], dim=1)), l_action[:,i])	# add the ce_loss of the remaining actions

		return log_prob								

# goal idxs will be removed
class LMP(nn.Module):
	def __init__(self, latent_dim, state_dim, action_dim, hidden_dims, tanh=False, latent_reg=0.0, ar=False, ar_params=None, rnn_layers=4, goal_idxs=None, act_fn=nn.ReLU()):
		'''
			input: 
				- latent_dim: dimension of latent vector 
				- state_dim: dimension of state vector
				- act_dim: dimension of the action (onehot) vector
				- hidden_dims: array of hidden dimensions for the Primitive Prior, and Primitive policy (decoder) (if not AR)
				- tanh: if True, apply tanh activation to the latent
				- latent_reg: indicates the degree of p2 regularization that is applied to the latent vector
				- ar: if True, use the more convoluted PolicyNetwork: ARPolicyNetwork
				- ar_params, goal_idxs: never used
				- rnn_layers: number of layers the rnn has (use GRU)
					- recall: 
						GRU overcomes gradient vanishing problem in RNN
						Only has hidden state (like RNN), rather than hidden state + cell state (like LSTM)
						comparable performance to LSTM
				- act_fn: activation function used in the network
			purpose:
				- create:
					1. An encoder to: encode the primitive
					2. A decoder to decode the primitive: primitive decoder
					3. A prior to bias the primitive embedding 
		'''
		super(LMP, self).__init__()
		self.latent_dim = latent_dim
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.hidden_dims = hidden_dims
		self.rnn_layers = rnn_layers
		self.tanh = tanh
		self.latent_reg = latent_reg
		self.act_fn = act_fn
		self.ar = ar
		self.create_encoder()
		# note: the dimension of the decoder return is different for each setting
		if self.ar:						# use a convoluted way to decode the primitive policy; 					note: forward() return 1 thing: the array(action)
			self.decoder = ARPolicyNetwork(latent_dim=latent_dim, state_dim=self.state_dim, act_dim=action_dim, act_fn=act_fn) 
		else:							# use a more simplestic way to decode the primitive policy; 			note: forward() return 2 things: array(mean), array(std)
			self.decoder = PolicyNetwork(latent_dim=latent_dim, state_dim=self.state_dim, hidden_dims=hidden_dims, act_dim=action_dim, act_fn=act_fn)
		
		# prior: over the primitive embedding (state -> latent_dim). Network will generate the mean and std over this space
		self.prior = PolicyNetwork(latent_dim=0, state_dim=self.state_dim, hidden_dims=hidden_dims, act_dim=latent_dim, act_fn=act_fn)

	def create_encoder(self):
		''' 
			purpose:
				- create a series of encoders that are used to encode the trajectories
		'''
		self.state_encoder = FCNetwork(inp_dim=self.state_dim, hidden_dims=self.hidden_dims, out_dim=self.hidden_dims[-1], act_fn=self.act_fn)				# (state -> state_embedding) dim(state_embedding) = self.hidden_dims[-1]
		#GRU: input = state_embedding + action, hidden size = state_embedding 
			# set to be a bidirectional GRU (GRU for forward and backward direction)
			# each GRU unit output is the same as that unit's hidden state
			# batch_first -> input tensor of shape (batch, seq, input_dim(state_dim))
		self.birnn_encoder = nn.GRU(self.hidden_dims[-1] + self.action_dim, self.hidden_dims[-1], self.rnn_layers, batch_first=True, bidirectional=True)	# (state_embedding + action -> state_embedding)
		# mean_encoder and logstd_encoder are 1 layer linnear network (with no nonlinearities)
		self.mean_encoder = nn.Linear(2*self.hidden_dims[-1], self.latent_dim)							# (2*state_embedding -> latent_dim) for mean; 		2 *state_embedding because we concatenate output of both backward and forward direction
		self.logstd_encoder = nn.Linear(2*self.hidden_dims[-1], self.latent_dim)						# (2*state_embedding -> latent_dim) for logstd

	def forward_encoder(self, state_traj, action_traj):
		'''
			input:
				(where si are state representation vector dim = state dim, and ai are action representation vector dim = action_dim)
				- state_traj of shape: [[<T1> s0, s1, s2...],
										[<T2> s0, s1, s2...],
										...					
										[<Tn> s0, s1, s2...]]
				- action_traj of shape: [[<T1> a0, a1, a2...],
										[<T2> a0, a1, a2...],
										...					
										[<Tn> a0, a1, a2...]]
			steps: 
				1. convert to state embedding
				2. combine state embedding with action
				3. pass through biGRU (bidirection GRU)
				4. combine the output of backward and forward direction (has all the information of the trajectories)
				5. use this to generate the mean and std for the primitive latent embedding for unsupervised learning
			output:
				- (tensor(mean), tensor(std)) - 1 tensor (mean and std) for each trajectory
		'''
		batch_size, seq_len = state_traj.size(0), state_traj.size(1)				# obtain the number of trajectories (batch_size) and number of states in the trajectory (seq_len)
		state_traj = state_traj.view(-1, self.state_dim)							# merge all states in all trajectories
		state_traj = self.state_encoder(state_traj)									# encode all states of all trajectories into state_embddings 
		state_traj = state_traj.view(batch_size, seq_len, -1)						# convert the tensor in the state_embedding space into batch_first format to input to GRU
		inp = torch.cat([state_traj, action_traj], dim=2)							# prepare input for GRU by embedding the state and action embedding
		out_birnn, _ = self.birnn_encoder(inp)
		#for each trajectory, obtain and concatenate 2 things:
			#1. the last output(hidden state) of the forward direction (after the last unit)
			#2. the last output(hidden state) of the backward direction (after the first unit)
		h = torch.cat([out_birnn[:, -1, :self.hidden_dims[-1]], out_birnn[:, 0, self.hidden_dims[-1]:]], dim=1) 
		
		# encode mean and std using linear network (Ax + b) to be simplistic because that h has all the information necessary to deduce the mean and std of the primitive embedding
		mean = self.mean_encoder(h)
		log_std = self.logstd_encoder(h)		#me: log_std so there is less error due to stretch in value(reduce range of value) #TODO: ask Dr.Lee
		return mean, log_std.exp()

	def calc_loss(self, state_traj, action_traj, is_cuda):
		'''
			input:
				(where si are state representation vector dim = state dim)
				- state_traj of shape: [[<T1> s0, s1, s2...],
										[<T2> s0, s1, s2...],
										...					
										[<Tn> s0, s1, s2...]]
				- action_traj of shape: [[<T1> a0, a1, a2...],
										[<T2> a0, a1, a2...],
										...					
										[<Tn> a0, a1, a2...]]
				- is_cuda: have GPU or not
			output:
				- kl_loss, nll_loss
			purpose:
				- kl_loss: distance between the distribution predicted only using the first state (prior) to be close to the distribution generated by looking at the trajectory (encoder)
		'''
		# Assumes all traj is of equal length
		batch_size, seq_len = action_traj.size(0), action_traj.size(1)				# obtain the number of trajectories (batch_size) and number of states in the trajectory (seq_len)
		nll_loss, kl_loss = 0., 0.													# initialize the loss values

		encoder_mean, encoder_std = self.forward_encoder(state_traj, action_traj)	# used forward() to predict the primitive embedding's mean and variance using the ENTIRE trajectory
		prior_mean, prior_std = self.prior(latent=None, state=state_traj[:,0,:])	# use Prior Network (self.prior) to predict the prior over the latent vector space, using ONLY the FIRST state of the trajectory
		
		# calculate KL distance between the predicted distribution and the prior distribution, want the distribution predicted only using the first state (prior) to be close to the distribution generated by looking at the trajectory (encoder)
		# TODO: ask Dr.Lee: why do we want these 2 distributions to be close to one another? (we will use the self.prior to select the primitive when learning with OPAL)
		kl_loss = kld_gauss(encoder_mean, encoder_std, prior_mean, prior_std)		

		zeros = torch.zeros(batch_size, self.latent_dim)
		ones = torch.ones(batch_size, self.latent_dim)
		if is_cuda:
			zeros = zeros.cuda()
			ones = ones.cuda()

		eps_dist = torch.distributions.Normal(zeros, ones)				# create a normal distribution of mean 0, variance 1 
		latent = encoder_mean + eps_dist.sample()*encoder_std			# add random noise of mean 0, variance 1 to the encoder, scaled by the encoder's std (encoder_std)
																		# TODO: ask Dr.Lee: why do we add noise like this, but not sample from Gaussian distribution with mean and std as defined by the encoder? to be able to pass gradient

		if self.tanh:
			latent = torch.tanh(latent)									# TODO: ask Dr.Lee: use tanh here to make sure that the latent is restricted to -1 to 1? why? maybe best practice

		for t in range(seq_len):
			nll_loss -= self.decoder.calc_log_prob(latent=latent, state=state_traj[:,t], action=action_traj[:,t])	# use the decoder to see how likely it is for such latent embedding and state will result in the action taken at that time step

		nll_loss = nll_loss/seq_len										#normalized over all actions in the sequence

		if self.latent_reg > 0:
			nll_loss += self.latent_reg*latent.norm(p=2, dim=1).mean()			#regularize the latent variable, if desired	 

		return kl_loss, nll_loss


A = LMP(2,2,2,[2,2], ar=True)
B = A.forward_encoder(torch.ones(2,2,2), torch.ones(2,2,2))
C = A.decoder.forward(torch.ones(1,2), torch.ones(1,2))
C = A.calc_loss(torch.ones(2,2,2), 0.5 * torch.ones(2,2,2), False)
print('hi')

