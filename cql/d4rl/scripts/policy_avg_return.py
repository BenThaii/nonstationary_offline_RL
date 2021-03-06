from rlkit.samplers.rollout_functions import rollout, multitask_rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
from rlkit.core import eval_util, logger
import gym



def get_policy_average_returns(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    # env = data['evaluation/env']
    env = gym.make('antmaze-medium-diverse-v0')
    env.seed(100)

    print("Policy loaded")

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    eval_paths = []
    for i in range(100):
        path = rollout(
                env,
                policy)
        # print(path['rewards'])
        eval_paths.append(path)
        print(i)
    logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--file', type=str, default="model_to_evaluate/itr_2700_myfix.pkl",
    # parser.add_argument('--file', type=str, default="model_to_evaluate/antmaze_pad_3000_mse/itr_2700.pkl", help='path to the snapshot file')
    parser.add_argument('--file', type=str, default="model_to_evaluate/antmaze_vanilla_3000/itr_2700.pkl", help='path to the snapshot file')

    parser.add_argument('--gpu', action='store_true', default=True)       # set to true if this argument is encountered
    args = parser.parse_args()

    get_policy_average_returns(args)
