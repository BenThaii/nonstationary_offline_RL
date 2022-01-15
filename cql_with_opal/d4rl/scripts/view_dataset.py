import d4rl, gym

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Path, Arrow
import numpy as np

def plot_track(verts, ax, **kw_args):
    '''Plot followed track: verts is 2D array: x, y'''
    for xy0, xy1 in zip(verts[:-1], verts[1:]):
        patch = Arrow(*xy0, *(xy1 - xy0), **kw_args)
        ax.add_patch(patch)
    ax.relim()
    ax.autoscale_view()




filename = str(uuid.uuid4())


def collect_trjectory(args):
    start_ind = 1001*4
    env = gym.make(args.env)
    dataset = env.env.get_dataset()
    
    fig, ax = plt.subplots()
    verts = np.vstack([dataset['observations'][start_ind:start_ind+1000, 0], dataset['observations'][start_ind:start_ind+1000, 1]]).T
    
    for i in range(0, 1000, 10):
        plot_track(verts[i:i+2], ax, color='red', fill=True)

    ax.plot(dataset['observations'][start_ind:start_ind+1000, 0], dataset['observations'][start_ind:start_ind+1000, 1])
    ax.annotate("start", (dataset['observations'][start_ind, 0], dataset['observations'][start_ind, 1]), color='purple', fontsize = 15, fontweight = 15)
    ax.annotate("end", (dataset['observations'][start_ind+1000-1, 0], dataset['observations'][start_ind+1000-1, 1]), color='purple', fontsize = 15, fontweight = 15)
    ax.axis([0, 22, 0, 22])
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="antmaze-medium-diverse-v0",
                        help='name of environment')
    args = parser.parse_args()

    collect_trjectory(args)
