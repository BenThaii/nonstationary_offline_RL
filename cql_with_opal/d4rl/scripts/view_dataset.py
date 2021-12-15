import d4rl, gym

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())


def collect_trjectory(args):
    env = gym.make(args.env)
    generation_object = d4rl.sequence_dataset(env)
    dataset = next(generation_object)
    actions = dataset['actions']
    num_actions = actions.shape[0]
    o = env.reset()
    env.render()
    for i in range(num_actions):
        env.step(actions[i])
        env.render()




    # data = torch.load(args.file)
    # policy = data['evaluation/policy']
    # env = data['evaluation/env']
    # print("Policy loaded")
    # if args.gpu:
    #     set_gpu_mode(True)
    #     policy.cuda()
    # while True:
        # path = rollout(
    #         env,
    #         policy,
    #         max_path_length=args.H,
    #         render=True,
    #     )
    #     if hasattr(env, "log_diagnostics"):
    #         env.log_diagnostics([path])
    #     logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="antmaze-medium-play-v0",
                        help='name of environment')
    args = parser.parse_args()

    collect_trjectory(args)
