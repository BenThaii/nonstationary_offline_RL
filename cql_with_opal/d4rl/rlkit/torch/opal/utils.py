import torch
from datetime import datetime
import ast
import os
import gym
import errno
import numpy as np
from rlkit.torch import pytorch_util as ptu


#BEN: i implemented this to bridge the gap for nn_models
def kld_gauss(mu_q, std_q, mu_p, std_p):
    """Analytical KLD between 2 Gaussians."""
    qs2 = std_q**2 + 1e-16
    ps2 = std_p**2 + 1e-16
    
    return (qs2/ps2 + ((mu_q-mu_p)**2)/ps2 + torch.log(ps2/qs2) - 1.0).sum()*0.5

def load_config(config_file_name):
    '''loading config from .txt file'''
    with open(config_file_name, "r") as data:
        dictionary = ast.literal_eval(data.read())
        print(dictionary)
        return dictionary

def clean_and_makedirs(dir_name, exp_name, seed = 0):
    '''
        prepare directory for 
        save_dir: saving network weights
        eval_dir: logging evalulation statistics
        log_dir: for miscellaneous logging

        for now, use the same directory for everything
        '''
    exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M") # current date and time
    output_dir = os.path.join(dir_name, exp_name, exp_timestamp)
    # save_dir = os.path.join(dir_name, exp_name, exp_timestamp, "model")
    # eval_dir = os.path.join(dir_name, exp_name, exp_timestamp, "model")
    # log_dir = os.path.join(dir_name, exp_name, exp_timestamp, "model")
    try:
        os.makedirs(output_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise
    
    return output_dir,output_dir,output_dir

def get_traj_dataset(env, env_name, traj_length):
    if env is None:
        env = gym.make(env_name)
    dataset = env.get_dataset()

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True


    obs_ls = []
    action_ls = []

    obs_ = []
    action_ = []

    N = dataset['rewards'].shape[0]
    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        
        #check against max episode length of the environment
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        
        if done_bool or final_timestep:
            # current_trajectory is finished (either terminated, or reached the end of the episode -> immediate transition becomes rubbish) -> dont add to the trajectory
            

            #truncate the current trajectory into chunks with length = traj_length            
            seq_length = len(obs_)
            for start_ind in range(seq_length):
                end_ind = start_ind + traj_length
                if end_ind > seq_length:
                    break

                obs_ls.append(obs_[start_ind:end_ind])
                action_ls.append(action_[start_ind:end_ind])

            
                

            #reset for new trajectory
            episode_step = 0
            obs_ = []
            action_ = []
        
        obs_.append(obs)
        action_.append(action)
        episode_step += 1

    return np.array(obs_ls), np.array(action_ls)

def get_opal_dataset(env, traj_length, primitive_encoder, discount = 0.99, dataset=None, terminate_on_end=False, **kwargs):
    
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    # do not save gradient in primitive encoder
    with torch.no_grad():
        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True


        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []
        obs_traj_ = []
        action_traj_ = []

        obs_ls = []
        action_ls = []
        reward_ls = []
        terminal_ls = []

        N = dataset['rewards'].shape[0]
        episode_step = 0
        for i in range(N-1):
            obs = dataset['observations'][i].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            terminal_bool = bool(dataset['terminals'][i])
            
            obs_ls.append(obs)
            action_ls.append(action)
            reward_ls.append(reward)
            terminal_ls.append(terminal_bool)

            #check against max episode length of the environment
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)
            
            if (not terminate_on_end) and final_timestep:
                # current_trajectory is finished (either terminated, or reached the end of the episode -> immediate transition becomes rubbish) -> dont add to the trajectory
                
                #truncate the current trajectory into chunks with length = traj_length            
                seq_length = len(obs_ls)
                
                # TODO: delete
                # action_ls_torch = torch.tensor(action_ls).to(ptu.device) 
                # obs_ls_torch = torch.tensor(obs_ls).to(ptu.device) 

                num_sub_traj = len(action_ls)+1-traj_length

                if seq_length > traj_length:        # have more than 1 sub trajectory in the sequence
                    action_subtraj_batch = []
                    obs_subtraj_batch = []
                    for start_ind in range(num_sub_traj):
                        end_ind = start_ind + traj_length

                        obs_subtraj = obs_ls[start_ind:end_ind]
                        action_subtraj = action_ls[start_ind:end_ind]
                        reward_subtraj = reward_ls[start_ind:end_ind]
                        terminal_subtraj = terminal_ls[start_ind:end_ind]

                        action_subtraj_batch.append(action_subtraj)
                        obs_subtraj_batch.append(obs_subtraj)

                        # obtain discounted rewards of the current subtrajectory
                        accum_reward = 0
                        for j in range(traj_length):
                            accum_reward += discount**j * reward_subtraj[j]

                        # obtain the terminal status of the current subtrajectory, identify as terminal if has had any terminal flag = True during subtrajectory
                        terminal = np.any(terminal_subtraj)
                        
                        # obtain next observation
                        if end_ind >= seq_length:
                            # next observation lies in the next sequence/trajectory
                            next_obs = dataset['observations'][i+1].astype(np.float32)
                        else:
                            next_obs = obs_ls[end_ind]

                        obs_.append(obs_ls[start_ind])
                        next_obs_.append(next_obs)
                        # action_.append(batch_latent_encoding[start_ind].cpu.numpy())
                        reward_.append(accum_reward)          
                        done_.append(terminal)
                        obs_traj_.append(obs_subtraj)
                        action_traj_.append(action_subtraj)
                        
                    obs_subtraj_batch = torch.Tensor(obs_subtraj_batch).to(ptu.device) 
                    action_subtraj_batch = torch.Tensor(action_subtraj_batch).to(ptu.device) 
                    batch_latent_encoding, _ = primitive_encoder(obs_subtraj_batch, action_subtraj_batch)
                    action_.extend(batch_latent_encoding.cpu().numpy())
                
                episode_step  = 0
                obs_ls = []
                action_ls = []
                reward_ls = []
                terminal_ls = []
            
            episode_step += 1
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'rewards': np.array(reward_),
            'next_observations': np.array(next_obs_),
            'terminals': np.array(done_),
            'obs_traj': np.array(obs_traj_),
            'action_traj': np.array(action_traj_),
        }



def old_dataset(env, traj_length, primitive_encoder, dataset=None, terminate_on_end=False, **kwargs):
    
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    # do not save gradient in primitive encoder
    with torch.no_grad():
        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True


        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []
        obs_traj_ = []
        action_traj_ = []

        obs_traj = []
        action_traj = []
        reward_traj = []

        N = 10
        N = dataset['rewards'].shape[0]
        episode_step = 0
        traj_step = 0
        for i in range(N-1):
            obs = dataset['observations'][i].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])
            
            #check against max episode length of the environment
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)
            
            if (not terminate_on_end) and (done_bool or final_timestep):
                # current_trajectory is finished (either terminated, or reached the end of the episode -> immediate transition becomes rubbish) -> dont add to the trajectory
                episode_step  = 0
                obs_traj = []
                action_traj = []
                reward_traj = []
                traj_step = -1  # because will be incremented at the end of the loop

            if traj_step == 0:
                if episode_step == 0:
                    prev_obs = obs     # first observation of the episode, has no prev_obs
                elif len(obs_traj) == traj_length:                  # only collect subtrajectories of the same length
                    # reach the start state of the (sub)trajectory with at least 1 trajectory before it (has prev_obs)
                    obs_.append(prev_obs)

                    
                    # obs_nparray = np.array(obs_traj)
                    # obs_torcharray = ptu.from_numpy(obs_nparray).float()
                    obs_torcharray = torch.tensor(obs_traj).to(ptu.device)   
                    obs_torcharray = obs_torcharray[None,:,:]             # add 1 dimension for batch size = 1

                    # action_nparray = np.array(action_traj)
                    # action_torcharray = ptu.from_numpy(action_nparray).float()
                    action_torcharray = torch.tensor(action_traj).to(ptu.device) 
                    action_torcharray = action_torcharray[None,:,:]             # add 1 dimension for batch size = 1

                    latent_encoding,_ = primitive_encoder(obs_torcharray, action_torcharray)    #only take the encoder mean (assume deterministic)

                    action_.append(latent_encoding.squeeze(0).cpu().numpy())     #TODO: may be incorrect
                    reward_.append(sum(reward_traj))
                    next_obs_.append(obs)
                    done_.append(done_bool)             # very rare that we see done_bool = True, because done_bool may happen in the middle of the (sub)trajectory
                    
                    
                    obs_traj_.append(np.array(obs_traj))
                    action_traj_.append(np.array(action_traj))
                    if np.array(obs_traj_).dtype == np.dtype('O'):
                        print('hi')
                    prev_obs = obs

                    obs_traj = []
                    action_traj = []
                    reward_traj = []



            obs_traj.append(obs)
            action_traj.append(action)
            reward_traj.append(reward)
            episode_step += 1
            traj_step = (traj_step + 1)%traj_length
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'rewards': np.array(reward_),
            'next_observations': np.array(next_obs_),
            'terminals': np.array(done_),
            'obs_traj': np.array(obs_traj_),
            'action_traj': np.array(action_traj_),
        }