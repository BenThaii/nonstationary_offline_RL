from rlkit.samplers.rollout_functions import nonstationary_rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
import copy
from rlkit.torch.pad.networks import nonstationary_rollout_pad
from grounding_codes.atp_envs import *
import gym

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    # env = data['evaluation/env']
    env = gym.make(args.eval_env)
    # env = gym.wrappers.Monitor(env, "~/.Videos",force=True)


    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    while True:
        cur_policy = copy.deepcopy(policy)
        path = nonstationary_rollout_pad(
            env,
            cur_policy,
            use_pad = args.use_pad,
            max_path_length=args.H,
            render=True,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')       # set to true if this argument is encountered
    # parser.add_argument("--eval_env", type=str, default='HalfCheetahModified-v2')
    parser.add_argument("--eval_env", type=str, default='HalfCheetahBroken-v2')
    # parser.add_argument("--eval_env", type=str, default='antmaze-medium-diverse-v0')
    parser.add_argument("--use_pad", action='store_true')

    args = parser.parse_args()

    simulate_policy(args)

