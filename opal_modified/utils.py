import torch
from datetime import datetime
import ast
import os
import gym

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
    os.makedirs(output_dir)
    return output_dir,output_dir,output_dir

def get_traj_dataset(env, env_namme, traj_length):
    return

