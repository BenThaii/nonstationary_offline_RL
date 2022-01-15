import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBufferOpal
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.opal.wrappers import LatentPolicyEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql_opal_raw import CQLTrainer
# from rlkit.torch.sac.cql_opal import CQLTrainer
from rlkit.torch.opal.nn_models import LMP
from rlkit.torch.opal.utils import load_config, get_opal_dataset
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from datetime import datetime
import torch

import argparse, os
import numpy as np

import h5py
import d4rl, gym

def load_hdf5_opal(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    # Center reward for Ant-Maze
    replay_buffer._rewards = (np.expand_dims(dataset['rewards'], 1) - 0.5)*4.0   
    replay_buffer._terminals = np.expand_dims(dataset['terminals'], 1)  
    replay_buffer._size = dataset['terminals'].shape[0]
    
    replay_buffer._obs_traj = dataset['obs_traj']
    replay_buffer._action_traj = dataset['action_traj']

    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size

def experiment(variant):
    env = gym.make(variant['env_name'])

    if variant['continue']:
        # continue training
        
        data = torch.load(variant['cont_pkl_file'])

        # loading opal primitive configurations and networks
        config = load_config(variant['opal_env_config'])
        # DEFAULT PARAMS for opal configuration
        if 'lr' not in config:
            config['lr'] = 1e-3
        if 'weight_decay' not in config:
            config['weight_decay'] = 0
        if 'latent_reg' not in config:
            config['latent_reg'] = 0
        if 'ar' not in config:
            config['ar'] = False
        opal_policy = data['trainer/opal_unsupervised_policy']
        opal_policy.to(ptu.device)                              #convert the opal network to the same device as everything else
        primitive_decoder_opal = opal_policy.decoder
        primitive_encoder = opal_policy.forward_encoder
        prior_encoder = opal_policy.prior

        eval_env = LatentPolicyEnv(env, primitive_decoder_opal, config['latent_dim'], 
                                    config['traj_length'], 
                                    prior_encoder
                                    )
        expl_env = eval_env

        qf1 = data['trainer/qf1']
        qf2 = data['trainer/qf2']
        target_qf1 = data['trainer/target_qf1']
        target_qf2 = data['trainer/target_qf2']
        policy = data['trainer/policy']
        
    else: 
        # train from scratch


        # loading opal primitive configurations and networks
        config = load_config(variant['opal_env_config'])
        # DEFAULT PARAMS for opal configuration
        if 'lr' not in config:
            config['lr'] = 1e-3
        if 'weight_decay' not in config:
            config['weight_decay'] = 0
        if 'latent_reg' not in config:
            config['latent_reg'] = 0
        if 'ar' not in config:
            config['ar'] = False
        opal_policy = LMP(latent_dim=config['latent_dim'], state_dim=env.observation_space.shape[0], 
		    action_dim=env.action_space.shape[0], hidden_dims=config['hidden_dims'], goal_idxs=config['goal_idxs'],
            tanh=config['tanh'], latent_reg=config['latent_reg'], ar=config['ar'])
        
        checkpoint = torch.load(variant['opal_primitive_file'])
        opal_policy.load_state_dict(checkpoint['gp_aa_model'])
        opal_policy.to(ptu.device)                              #convert the opal network to the same device as everything else
        primitive_decoder_opal = opal_policy.decoder
        primitive_encoder = opal_policy.forward_encoder
        prior_encoder = opal_policy.prior
        

        eval_env = LatentPolicyEnv(env, primitive_decoder_opal, config['latent_dim'], config['traj_length'], prior_encoder)
        expl_env = eval_env

        obs_dim = expl_env.observation_space.low.size
        action_dim = config['latent_dim']

        M = variant['layer_size']
        qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        target_qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        target_qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        # tanh Gaussian policy is just a Gaussian policy , whose selected action is squashed through a tanh
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M, M], 
        )

        



    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    buffer_filename = None
    if variant['buffer_filename'] != None:
        buffer_filename = variant['buffer_filename']
    
    replay_buffer = EnvReplayBufferOpal(
        max_replay_buffer_size = variant['replay_buffer_size'],
        env = expl_env,
        primitive_traj_length = config['traj_length'],
    )
    if variant['load_buffer'] and buffer_filename != None:
        replay_buffer.load_buffer(buffer_filename)
    else:
        if variant['latent_dataset']:
            dataset = np.load(variant['latent_dataset'], allow_pickle = True)
            dataset = dataset[()]
        else:
            dataset = get_opal_dataset(eval_env, traj_length= config['traj_length'], primitive_encoder= primitive_encoder, discount=variant['trainer_kwargs']['discount'])
        load_hdf5_opal(dataset, replay_buffer)
       
    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        latent_policy= primitive_decoder_opal,
        opal_unsupervised_policy = opal_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

def enable_gpus(gpu_str):
    if (gpu_str != ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        algorithm_kwargs=dict(
            num_epochs=3001,
            # num_eval_steps_per_epoch=1000,
            # num_trains_per_train_loop=1000,  
            # num_expl_steps_per_train_loop=1000,
            # min_num_steps_before_training=1000,
            # max_path_length=1000,

            num_eval_steps_per_epoch=2000,
            num_trains_per_train_loop=2000,  
            num_expl_steps_per_train_loop=2000,
            min_num_steps_before_training=2000,
            max_path_length=2000,

            batch_size=128,
            # batch_size=512,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            # policy_lr=1E-4,
            # qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            # use_automatic_entropy_tuning=False,

            # Target nets/ policy vs Q-function update
            cql_start=40000,        #policy_eval_start

            # CQL
            cql_temp=1.0,
            version=3,                          #min_q_version
            cql_alpha=5.0,                      #min_q_weight

            # lagrange
            use_cql_alpha_tuning=True,          # with_lagrange
            # use_cql_alpha_tuning=False,          # with_lagrange
            

            # opal stuff:
            only_nll_before_start = False,          # for behavior cloning of latent variable
            latent_policy_train = False,              # for behavior cloning of primitive decoder           


            # suggested in the paper:
            policy_lr = 3E-5,
            qf_lr = 3E-4,
            latent_policy_lr = 3E-4,
            cql_tau=5.0,     
            cql_alpha_min = 0.01,     
            min_q_weight = 5
        )
    )
    




    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='antmaze-medium-play-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--max_q_backup", type=str, default="False")          # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic_backup", type=str, default="True")   # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--policy_eval_start", default=40000, type=int)       # Defaulted to 20000 (40000 or 10000 work similarly)
    # parser.add_argument("--policy_eval_start", default=40, type=int)       # Defaulted to 20000 (40000 or 10000 work similarly)
    
    parser.add_argument('--policy_lr', default=3E-5, type=float)              # Policy learning rate
    parser.add_argument('--version', default=3, type=int)               # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho)) 
    parser.add_argument('--lagrange_thresh', default=5.0, type=float)         # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--opal_primitive_file', type =str)
    parser.add_argument('--opal_env_config', type =str)
    parser.add_argument('--latent_dataset', type =str)
    parser.add_argument('--min_q_weight', default=5.0, type=float)           


    args = parser.parse_args()
    # enable_gpus(args.gpu)
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['version'] = args.version
    variant['trainer_kwargs']['cql_temp'] = 1.0
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['cql_start'] = args.policy_eval_start
    variant['trainer_kwargs']['cql_tau'] = args.lagrange_thresh
    if args.lagrange_thresh < 0.0:
        variant['trainer_kwargs']['use_cql_alpha_tuning'] = False
    
    variant['buffer_filename'] = None

    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = args.seed

    variant['opal_primitive_file'] = "opal_primitive/" + args.opal_primitive_file
    variant['opal_env_config'] = "opal_config/" + args.opal_env_config
    variant['latent_dataset'] = args.latent_dataset



    variant['continue'] = False
     
    # # continue policy training code
    # variant['continue'] = True
    # variant['cont_pkl_file'] = "/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211207-1111-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211207_1111_2021_12_07_11_11_11_0000--s-0/itr_3000.pkl"
    # # continue policy training code ends...

    start_time = datetime.now().strftime("%Y%m%d_%H%M") # current date and time

    setup_logger(os.path.join('CQL_offline_mujoco_runs', args.env + "_" + start_time), snapshot_mode="gap", snapshot_gap= 50, variant=variant, 
                                    base_log_dir='/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/')
    ptu.set_gpu_mode(True)
    experiment(variant)