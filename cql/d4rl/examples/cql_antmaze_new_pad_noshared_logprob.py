import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.pad.networks import EncodedQF, EncodedTanhGaussianPolicy, MdpPathCollector_pad, CustomMDPPathCollector_pad
# from rlkit.torch.sac.cql import CQLTrainer
# from rlkit.torch.sac.cql_myfix import CQLTrainer
# from rlkit.torch.sac.cql_policy_update_first_contributor import CQLTrainer
# from rlkit.torch.sac.cql_pad import CQLTrainer
from rlkit.torch.sac.cql_pad_noshared import CQLTrainer
# from rlkit.torch.sac.cql_pad_v2 import CQLTrainer
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from datetime import datetime
from torch.nn import functional as F



import argparse, os
import numpy as np

import h5py
import d4rl, gym

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    # Center reward for Ant-Maze
    replay_buffer._rewards = (np.expand_dims(dataset['rewards'], 1) - 0.5)*4.0   
    replay_buffer._terminals = np.expand_dims(dataset['terminals'], 1)  
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size

def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env
    
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    encoder_layer_size = variant['encoder_layer_size']
    encoder_num_layers = variant['encoder_num_layers']
    encoder_hidden_sizes = [encoder_layer_size for i in range(encoder_num_layers)]

    inv_num_layers = variant['inv_num_layers']
    inv_hidden_sizes = [M for i in range(inv_num_layers)]

    qf_num_layers = variant['qf_num_layers']
    qf_hidden_sizes = [M for i in range(qf_num_layers)]

    policy_num_layers = variant['policy_num_layers']
    policy_hidden_sizes = [M for i in range(policy_num_layers)]

    encoder_feature_dim = variant['encoder_feature_dim']

    # encoder_hidden_sizes = []
    # inv_hidden_sizes = []
    # cql_hidden_sizes = [M, M]

    # encoder_hidden_sizes = [M, M, M, M]
    # inv_hidden_sizes = [M, M, M, M]
    # cql_hidden_sizes = [M, M, M, M, M]

    

    state_encoder = Mlp(
        input_size=obs_dim,
        output_size=encoder_feature_dim,
        hidden_sizes= encoder_hidden_sizes,
        output_activation=F.relu,                   #Ben: rather than using identity at last layer
    )

    if variant['inv_loss_type'] == "mse":
        inv_network = FlattenMlp(
            input_size= 2 * encoder_feature_dim,        #for both current, and next observation
            output_size= action_dim, 
            hidden_sizes= inv_hidden_sizes, 
        )
    elif variant['inv_loss_type'] == "variational":
        inv_network = FlattenMlp(
            input_size= 2 * encoder_feature_dim,        #for both current, and next observation
            output_size= 2 * action_dim, 
            hidden_sizes= inv_hidden_sizes, 
        )
    else:
        raise Exception('can only accept "mse" or "variational" inv_loss_type')

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=qf_hidden_sizes,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=qf_hidden_sizes,
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=qf_hidden_sizes,
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=qf_hidden_sizes,
    )


    policy = EncodedTanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=policy_hidden_sizes, 
        encoder = state_encoder,
        inv_network = inv_network,
        encoder_lr= variant['encoder_lr'],
        inv_lr= variant['inv_lr'],
        inv_loss_type = variant['inv_loss_type']
    )

    print(count_parameters(policy) - count_parameters(policy.inv_network))


    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector_pad(
        env=eval_env,
        policy=eval_policy,
        use_pad=variant['trainer_kwargs']['use_pad_inv_loss']
    )
    expl_path_collector = CustomMDPPathCollector_pad(
        env=eval_env,
        policy=policy,
        qf=qf1,
        use_pad=variant['trainer_kwargs']['use_pad_inv_loss']
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']
    
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    else:
        load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)
       
    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
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
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        encoder_layer_size=256,
        encoder_feature_dim=256,
        encoder_num_layers = 1,
        qf_num_layers = 3,
        policy_num_layers = 1,
        inv_num_layers = 2,

        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        algorithm_kwargs=dict(
            num_epochs=3001,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,  
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        #PAD,
        encoder_lr = 1e-5,
        inv_lr = 1e-5,
        inv_loss_type = "mse",
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=40000,
            num_qs=2,

            # CQL
            temp=1.0,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=True,   # Defaults to true
            lagrange_thresh=10.0,
            
            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,

            # PAD:
            use_pad_inv_loss = True,
            inv_loss_start = 40000,  
            inv_interval = 1,      
        ),
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='HalfCheetah-v2')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--max_q_backup", type=str, default="False")          # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic_backup", type=str, default="True")   # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--policy_eval_start", default=40000, type=int)       # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument('--min_q_weight', default=1.0, type=float)            # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--policy_lr', default=1e-4, type=float)              # Policy learning rate
    parser.add_argument('--min_q_version', default=3, type=int)               # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho)) 
    parser.add_argument('--lagrange_thresh', default=5.0, type=float)         # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['temp'] = 1.0
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    if args.lagrange_thresh < 0.0:
        variant['trainer_kwargs']['with_lagrange'] = False
    
    variant['buffer_filename'] = None

    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = args.seed

    rnd = np.random.randint(0, 1000000)
    start_time = datetime.now().strftime("%Y%m%d_%H%M") # current date and time
    setup_logger(os.path.join('CQL_offline_mujoco_runs', args.env + "_" + start_time), snapshot_mode="gap", snapshot_gap= 300, variant=variant, 
                                    base_log_dir='/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/')
    ptu.set_gpu_mode(True)
    experiment(variant)
