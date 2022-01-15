import torch
from datetime import datetime
import ast
import os
import gym
import errno
import numpy as np

# #####BEN: I implemented this to bridge the gap for nn_models
def kld_gauss(mu_q, std_q, mu_p, std_p):
    """Analytical KLD between 2 Gaussians."""
    qs2 = std_q**2 + 1e-16
    ps2 = std_p**2 + 1e-16
    
    return (qs2/ps2 + ((mu_q-mu_p)**2)/ps2 + torch.log(ps2/qs2) - 1.0).sum()*0.5

# def kld_gauss(mean_1, std_1, mean_2, std_2):
#     kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
#         (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
#         std_2.pow(2) - 1)
#     return 0.5 * torch.sum(kld_element)

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
    # number transitions in the dataset
    N = dataset['rewards'].shape[0]
    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        
        #check against max episode length of the environment
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        
        # if done_bool or final_timestep:
        if final_timestep:
            # current_trajectory is finished (either terminated, or reached the end of the episode -> immediate transition becomes rubbish) -> dont add to the trajectory
            

            #truncate the current trajectory into sections with length = traj_length            
            seq_length = len(obs_)
            

            for start_ind in range(seq_length):
                end_ind = start_ind + traj_length
                if end_ind > seq_length:
                    break
                # end_ind = min(end_ind, seq_length)      # the final section may not contain as many transitions as the previous sections

                obs_ls.append(obs_[start_ind:end_ind])
                action_ls.append(action_[start_ind:end_ind])



            #reset for new trajectory
            episode_step = 0
            obs_ = []
            action_ = []
        
        obs_.append(obs)
        action_.append(action)
        episode_step += 1


    # shuffle both datasets (not yet handled in gp_aa)
    obs_ls_np = np.array(obs_ls)
    action_ls_np = np.array(action_ls)
    p = np.random.permutation(obs_ls_np.shape[0])

    return obs_ls_np[p], action_ls_np[p]

