from rlkit.samplers.rollout_functions import rollout, multitask_rollout
from rlkit.torch.pad.networks import nonstationary_rollout_pad
import copy
from grounding_codes.atp_envs import *

from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
from rlkit.core import eval_util, logger
# import gym



def get_policy_average_returns(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    # env = data['evaluation/env']
    env = gym.make(args.eval_env)
    env.seed(100)

    print("Policy loaded")

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    eval_paths = []
    for _ in range(10):
        cur_policy = copy.deepcopy(policy)
        path = nonstationary_rollout_pad(
            env,
            cur_policy,
            use_pad = args.use_pad,
            variation_attribute = 'dof_friction',
            variation_type = 'linear-increase',
            hasVariation = True,
            variation_amplitude = 5,
        )
        # print(path['rewards'])
        eval_paths.append(path)
    
    logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--file', type=str, default="model_to_evaluate/itr_2700_myfix.pkl",
    parser.add_argument('--file', type=str, default="model_to_evaluate/itr_2700_contributor.pkl",
                        help='path to the snapshot file')

    parser.add_argument('--gpu', action='store_true')       # set to true if this argument is encountered
    # parser.add_argument("--eval_env", type=str, default='HalfCheetahModified-v2')
    parser.add_argument("--eval_env", type=str, default='HalfCheetah-v2')
    parser.add_argument("--use_pad", action='store_true')
    args = parser.parse_args()

    get_policy_average_returns(args)
