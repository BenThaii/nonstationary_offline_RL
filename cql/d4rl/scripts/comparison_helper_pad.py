from email import policy
from grounding_codes.atp_envs import *

from rlkit.torch.pytorch_util import set_gpu_mode
import torch
from rlkit.core import eval_util        
from rlkit.torch.pad.networks import nonstationary_rollout_pad
from textwrap import wrap

import time
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import pandas as pd



def plot_comparison_old(excel_filepath, last_experiment = False):

    excel_file_name = os.path.split(excel_filepath)[-1][:-5]
    excel_parent_dir = os.path.relpath(os.path.join(excel_filepath, os.pardir))
    figures_parent_dir = os.path.join(excel_parent_dir, 'figures')
    os.makedirs(figures_parent_dir, exist_ok=True)

    min_max_plot = False
    name_dict = {}
    name_dict['vanilla'] = {True: "vanilla cql", False: "vanilla cql"}
    name_dict['pad'] = {True: "cql with pad, deployment with ss", False: "cql with pad, deployment WITHOUT ss"}

    skip_dict = {}
    skip_dict['vanilla'] = {True: False, False: False}
    skip_dict['pad'] = {True: False, False: True}

    comparison_df = pd.read_excel(open(excel_filepath, 'rb'),
                sheet_name='raw_data')  

    metric_val_name = 'Returns Mean'
    metric_std_name = 'Returns Std'
    metric_min_name = 'Returns Min'
    metric_max_name = 'Returns Max'

    fig = plt.figure()
    ax = plt.gca()
    for policy_type in comparison_df['policy_type'].unique():
        for pad_type in comparison_df[comparison_df['policy_type'] == policy_type]['use_pad'].unique():
            

            instance_df = comparison_df[(comparison_df['policy_type'] == policy_type) & (comparison_df['use_pad'] == pad_type)]

            if 'vanilla' in policy_type:
                if skip_dict['vanilla'][pad_type]:  #check if we want to skip this plot or not
                    continue
                instance_name = name_dict['vanilla'][pad_type]
            elif 'pad' in policy_type:
                if skip_dict['pad'][pad_type]:      #check if we want to skip this plot or not
                    continue
                instance_name = name_dict['pad'][pad_type]
            else:
                raise Exception('can only accept vanilla, or pad policy type')

            mean_val = instance_df[metric_val_name]
            std_val = instance_df[metric_std_name]
            min_val = instance_df[metric_min_name]
            max_val = instance_df[metric_max_name]
            
            color = next(ax._get_lines.prop_cycler)['color']
            if min_max_plot:
                plt.errorbar(instance_df['epoch'], mean_val, [mean_val - min_val, max_val - mean_val], color = color, fmt='ok', lw=1)
            plt.errorbar(instance_df['epoch'], mean_val, yerr=std_val, marker = 'o', capsize=10, color = color, fmt='k', lw=3, label="{}_{}".format(policy_type, pad_type))

    plt.legend(loc='lower right')
    plt.title(os.path.split(excel_filepath)[-1])

    img_filepath = os.path.join(figures_parent_dir, '{}.png'.format(excel_file_name))
    plt.savefig(img_filepath)
    plt.pause(1)

    if last_experiment:
        plt.show()
    else:
        plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_comparison(excel_filepath, mean_only = False, last_experiment = False):
    excel_file_name = os.path.split(excel_filepath)[-1][31:-5]
    excel_parent_dir = os.path.relpath(os.path.join(excel_filepath, os.pardir))
    figures_parent_dir = os.path.join(excel_parent_dir, 'progression')
    os.makedirs(figures_parent_dir, exist_ok=True)

    min_max_plot = False
    name_dict = {}
    name_dict['vanilla'] = {True: "vanilla cql", False: "vanilla cql"}
    name_dict['pad'] = {True: "cql with pad, deployment with ss", False: "cql with pad, deployment WITHOUT ss"}

    skip_dict = {}
    skip_dict['vanilla'] = {True: False, False: False}
    skip_dict['pad'] = {True: False, False: True}

    comparison_df = pd.read_excel(open(excel_filepath, 'rb'),
                sheet_name='raw_data')  

    # Plotting by Epoch, then by solution type
    fig = plt.figure()
    ax = plt.subplot(111)

    plt_pos = 1
    start_idx = 1
    max_num_policies = 0

    x_label_vals = []
    x_label_pos_ls = []

    policy_name_dict = {}

    for epoch in sorted(comparison_df['epoch'].unique()):
        epoch_df = comparison_df[comparison_df['epoch'] == epoch]
        epoch_data_ls = []
        policy_num = 0    # for color generation9199
        start_idx = plt_pos
        # policy_types = ['hopper_pad_3000_mse_old', 'hopper_pad_3000_mse_invextralayer_old', 'hopper_pad_3000_variational_old', 'hopper_pad_3000_variational_invextralayer_old', 'hopper_vanilla_3000']
        # policy_types = ['antmaze_pad_3000_mse', 'antmaze_pad_3000_variational', 'antmaze_vanilla_3000']
        # for policy_type in policy_types:
        for policy_type in sorted(epoch_df['policy_type'].unique()):
            epoch_policy_df = epoch_df[epoch_df['policy_type'] == policy_type]
            for pad_type in sorted(epoch_policy_df['use_pad'].unique()):
                color = plt.cm.rainbow(policy_num*50)
                
                epoch_policy_pad_df = epoch_policy_df[epoch_policy_df['use_pad'] == pad_type]
                policy_name = '{}_{}'.format(policy_type, pad_type)
                policy_data = epoch_policy_pad_df['rewards']

                # plotting box plots
                if not mean_only:
                    bp = ax.boxplot(policy_data, 
                                positions = [plt_pos], 
                                widths = 0.6, 
                                showfliers = True,
                                )
                    # styling box plots
                    plt.setp(bp['boxes'], color = color)
                    plt.setp(bp['whiskers'], color = color)
                    plt.setp(bp['caps'], color = color)
                    plt.setp(bp['fliers'], color = color)
                    plt.setp(bp['medians'], color = 'r', linewidth=2)
                    
                # plotting mean
                if start_idx == 1:
                    # has label
                    ax.plot(plt_pos + 0.02 * policy_num, policy_data.mean(), 's', color= color, alpha=0.8, label='\n'.join(wrap(policy_name.replace("_old", "").replace("variational", "nll").replace("_True", "").replace("_False", "").replace("3000", "").replace("_", " "), 17)))
                else:
                    # does not need to have labels again
                    ax.plot(plt_pos + 0.02 * policy_num, policy_data.mean(), 's', color= color, alpha=0.8)
                

                # administrative stuff
                if not mean_only:
                    plt_pos += 1
                policy_num += 1
                    
                if policy_name not in policy_name_dict:
                    policy_name_dict[policy_name] = color
                else:
                    if policy_name_dict[policy_name] != color:
                        raise Exception('there is a color conflict. Reason: repeated policy_name, or different sets of policy names between epochs')

                epoch_data_ls.append(policy_data)
                if policy_num > max_num_policies:
                    max_num_policies = policy_num

        if not mean_only:
            epoch_label_pos = (start_idx + plt_pos-1)/2
        else:
            epoch_label_pos = start_idx
        x_label_vals.append("{}".format(epoch))
        x_label_pos_ls.append(epoch_label_pos)
        plt_pos += 1   # gap between epochs






    # labelling axes
    ax.set_xticks(x_label_pos_ls, x_label_vals)
    ax.set_xlabel('epoch')
    ax.set_ylabel('avg total returns')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])   # shrink main plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labelspacing = 2)            # display the legends

    plt.title(excel_file_name.replace("_old", "").replace("_", " "), x=0.7)

    img_filepath = os.path.join(figures_parent_dir, '{}.png'.format(excel_file_name))
    plt.savefig(img_filepath)
    plt.pause(1)

    if last_experiment:
        plt.show()
    else:
        plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_single_epoch_comparison_old(excel_filepath, epoch, last_experiment = False):

    excel_file_name = os.path.split(excel_filepath)[-1][14:-5]
    excel_parent_dir = os.path.relpath(os.path.join(excel_filepath, os.pardir))
    figures_parent_dir = os.path.join(excel_parent_dir, 'last_epoch')
    os.makedirs(figures_parent_dir, exist_ok=True)

    min_max_plot = False
    name_dict = {}
    name_dict['vanilla'] = {True: "vanilla cql", False: "vanilla cql"}
    name_dict['pad'] = {True: "cql with pad, deployment with ss", False: "cql with pad, deployment WITHOUT ss"}

    skip_dict = {}
    skip_dict['vanilla'] = {True: False, False: False}
    skip_dict['pad'] = {True: False, False: True}

    comparison_df = pd.read_excel(open(excel_filepath, 'rb'),
                sheet_name='raw_data')  

    metric_val_name = 'Returns Mean'
    metric_std_name = 'Returns Std'
    metric_min_name = 'Returns Min'
    metric_max_name = 'Returns Max'

    data_max = -np.inf
    data_min = np.inf

    fig = plt.figure()
    ax = plt.gca()
    for policy_type in comparison_df['policy_type'].unique():
        for pad_type in comparison_df[comparison_df['policy_type'] == policy_type]['use_pad'].unique():
            

            instance_df = comparison_df[(comparison_df['policy_type'] == policy_type) & (comparison_df['use_pad'] == pad_type)]
            instance_df = instance_df[instance_df['epoch'] == epoch]
            if 'vanilla' in policy_type:
                if skip_dict['vanilla'][pad_type]:  #check if we want to skip this plot or not
                    continue
                instance_name = name_dict['vanilla'][pad_type]
            elif 'pad' in policy_type:
                if skip_dict['pad'][pad_type]:      #check if we want to skip this plot or not
                    continue
                instance_name = name_dict['pad'][pad_type]
            else:
                raise Exception('can only accept vanilla, or pad policy type')

            mean_val = instance_df[metric_val_name]
            std_val = instance_df[metric_std_name]
            min_val = instance_df[metric_min_name]
            max_val = instance_df[metric_max_name]
            
            color = next(ax._get_lines.prop_cycler)['color']
            if min_max_plot:
                plt.errorbar(instance_df['epoch'], mean_val, [mean_val - min_val, max_val - mean_val], color = color, fmt='ok', lw=1)
            plt.errorbar(instance_df['epoch'], mean_val, yerr=std_val, color = color, fmt='k', marker = 'o', capsize=10, lw=3, label="{}_{}".format(policy_type, pad_type))

    plt.legend(loc='lower right')
    plt.title(os.path.split(excel_filepath)[-1])

    img_filepath = os.path.join(figures_parent_dir, '{}.png'.format(excel_file_name))
    plt.savefig(img_filepath)
    plt.pause(1)

    if last_experiment:
        plt.show()
    else:
        plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_single_epoch_comparison(excel_filepath, epoch, mean_only=False, last_experiment = False):

    excel_file_name = os.path.split(excel_filepath)[-1][31:-5]
    excel_parent_dir = os.path.relpath(os.path.join(excel_filepath, os.pardir))
    figures_parent_dir = os.path.join(excel_parent_dir, 'last_epoch')
    os.makedirs(figures_parent_dir, exist_ok=True)

    min_max_plot = False
    name_dict = {}
    name_dict['vanilla'] = {True: "vanilla cql", False: "vanilla cql"}
    name_dict['pad'] = {True: "cql with pad, deployment with ss", False: "cql with pad, deployment WITHOUT ss"}

    skip_dict = {}
    skip_dict['vanilla'] = {True: False, False: False}
    skip_dict['pad'] = {True: False, False: True}

    comparison_df = pd.read_excel(open(excel_filepath, 'rb'),
                sheet_name='raw_data')  

    # Plotting by Epoch, then by solution type
    fig = plt.figure()
    ax = plt.subplot(111)

    plt_pos = 1
    start_idx = 1
    max_num_policies = 0

    x_label_vals = []
    x_label_pos_ls = []

    policy_name_dict = {}

    
    epoch_df = comparison_df[comparison_df['epoch'] == epoch]
    epoch_data_ls = []
    policy_num = 0    # for color generation
    start_idx = plt_pos
    # policy_types = ['hopper_pad_3000_mse_old', 'hopper_pad_3000_mse_invextralayer_old', 'hopper_pad_3000_variational_old', 'hopper_pad_3000_variational_invextralayer_old', 'hopper_vanilla_3000']
    # policy_types = ['antmaze_pad_3000_mse', 'antmaze_pad_3000_variational', 'antmaze_vanilla_3000']

    # for policy_type in policy_types:
    for policy_type in sorted(epoch_df['policy_type'].unique()):
        epoch_policy_df = epoch_df[epoch_df['policy_type'] == policy_type]
        for pad_type in sorted(epoch_policy_df['use_pad'].unique()):
            color = plt.cm.rainbow(policy_num*50)
            
            epoch_policy_pad_df = epoch_policy_df[epoch_policy_df['use_pad'] == pad_type]
            policy_name = '{}_{}'.format(policy_type, pad_type)
            policy_data = epoch_policy_pad_df['rewards']

            # plotting box plots
            if not mean_only:
                bp = ax.boxplot(policy_data, 
                            positions = [plt_pos], 
                            widths = 0.6, 
                            showfliers = True,
                            )
                # styling box plots
                plt.setp(bp['boxes'], color = color)
                plt.setp(bp['whiskers'], color = color)
                plt.setp(bp['caps'], color = color)
                plt.setp(bp['fliers'], color = color)
                plt.setp(bp['medians'], color = 'r', linewidth=2)
                
            # plotting mean
            if start_idx == 1:
                # has label
                ax.plot(plt_pos, policy_data.mean(), 's', color= color, alpha=0.8, label='\n'.join(wrap(policy_name.replace("_old", "").replace("variational", "nll").replace("_True", "").replace("_False", "").replace("3000", "").replace("_", " "), 17)))
            else:
                # does not need to have labels again
                ax.plot(plt_pos, policy_data.mean(), 's', color= color, alpha=0.8)
            

            # administrative stuff
            plt_pos += 1
            policy_num += 1
                
            if policy_name not in policy_name_dict:
                policy_name_dict[policy_name] = color
            else:
                if policy_name_dict[policy_name] != color:
                    raise Exception('there is a color conflict. Reason: repeated policy_name, or different sets of policy names between epochs')

            epoch_data_ls.append(policy_data)
            if policy_num > max_num_policies:
                max_num_policies = policy_num

    epoch_label_pos = (start_idx + plt_pos-1)/2
    x_label_vals.append("{}".format(epoch))
    x_label_pos_ls.append(epoch_label_pos)
    plt_pos += 1   # gap between epochs



   
    

    # labelling axes
    ax.set_xticks(x_label_pos_ls, x_label_vals)
    ax.set_xlabel('epoch')
    ax.set_ylabel('avg total returns')
    ax.set_xlim(-2, plt_pos + 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])   # shrink main plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labelspacing = 2)            # display the legends
    
    # plt.subplots_adjust(left=0.1, right = 0.7)
    plt.title(excel_file_name.replace("_old", "").replace("_", " "), x=0.7)

    img_filepath = os.path.join(figures_parent_dir, '{}.png'.format(excel_file_name))
    plt.savefig(img_filepath)
    plt.pause(1)

    if last_experiment:
        plt.show()
    else:
        plt.show(block=False)
    plt.pause(1)
    plt.close()


def save_comparison_plots(excel_filepath):

    excel_file_name = os.path.split(excel_filepath)[-1][:-5]
    excel_parent_dir = os.path.relpath(os.path.join(excel_filepath, os.pardir))
    figures_parent_dir = os.path.join(excel_parent_dir, 'figures')
    os.makedirs(figures_parent_dir, exist_ok=True)

    min_max_plot = False
    name_dict = {}
    name_dict['vanilla'] = {True: "vanilla cql", False: "vanilla cql"}
    name_dict['pad'] = {True: "cql with pad, deployment with ss", False: "cql with pad, deployment WITHOUT ss"}

    skip_dict = {}
    skip_dict['vanilla'] = {True: False, False: False}
    skip_dict['pad'] = {True: False, False: True}

    comparison_df = pd.read_excel(open(excel_filepath, 'rb'),
                sheet_name='raw_data')  

    metric_val_name = 'Returns Mean'
    metric_std_name = 'Returns Std'
    metric_min_name = 'Returns Min'
    metric_max_name = 'Returns Max'

    fig = plt.figure()
    ax = plt.gca()
    for policy_type in comparison_df['policy_type'].unique():
        for pad_type in comparison_df[comparison_df['policy_type'] == policy_type]['use_pad'].unique():
            

            instance_df = comparison_df[(comparison_df['policy_type'] == policy_type) & (comparison_df['use_pad'] == pad_type)]

            if 'vanilla' in policy_type:
                if skip_dict['vanilla'][pad_type]:  #check if we want to skip this plot or not
                    continue
                instance_name = name_dict['vanilla'][pad_type]
            elif 'pad' in policy_type:
                if skip_dict['pad'][pad_type]:      #check if we want to skip this plot or not
                    continue
                instance_name = name_dict['pad'][pad_type]
            else:
                raise Exception('can only accept vanilla, or pad policy type')

            mean_val = instance_df[metric_val_name]
            std_val = instance_df[metric_std_name]
            min_val = instance_df[metric_min_name]
            max_val = instance_df[metric_max_name]
            
            color = next(ax._get_lines.prop_cycler)['color']
            if min_max_plot:
                plt.errorbar(instance_df['epoch'], mean_val, [mean_val - min_val, max_val - mean_val], color = color, fmt='ok', lw=1)
            plt.errorbar(instance_df['epoch'], mean_val, yerr=std_val, color = color, fmt='k', lw=3, label=instance_name)

    plt.legend(loc='lower right')
    plt.title(excel_file_name)

    img_filepath = os.path.join(figures_parent_dir, '{}.png'.format(excel_file_name))
    plt.savefig(img_filepath)
    plt.pause(1)




def get_policy_average_returns_per_epoch(filename, eval_env, hasVariation, seeds, use_gpu, use_pad, variation_attribute, variation_type, variation_amplitude, inv_interval = 5):
    data = torch.load(filename)
    policy = data['evaluation/policy']
    env = gym.make(eval_env)

    if use_gpu:
        set_gpu_mode(True)
        policy.cuda()

    eval_paths = []
    eval_rewards_ls = []
    for seed in seeds:
        env.seed(seed)
        cur_policy = copy.deepcopy(policy)
        path = nonstationary_rollout_pad(
            env,
            cur_policy,
            hasVariation = hasVariation,
            use_pad = use_pad,
            variation_attribute=variation_attribute,
            variation_type=variation_type,
            variation_amplitude=variation_amplitude,
            inv_interval = 5
        )
        eval_paths.append(path)
        eval_rewards_ls.append(sum(path['rewards']).item())
    return eval_rewards_ls
    return eval_util.get_generic_path_information(eval_paths)
    
def compare_policies_multi_epochs(
                                #   eval_env = 'HalfCheetahModified-v2',
                                  eval_env = 'HalfCheetah-v2',
                                #   hasVariation = False,
                                  hasVariation = True,
                                  variation_attribute = 'gravity',
                                #   variation_attribute = 'dof_friction',
                                  variation_type = 'linear-increase',
                                  variation_amplitude = 5,
                                  use_gpu = True,
                                  seeds = range(0,5), 
                                  filename_dirs = ['vanilla_3000', 'pad_3000', 'pad_3000'], 
                                  use_pads = [False, False, True], 
                                  epochs = range(0, 3001, 300),
                                  result_dir_name = "",
                                  last_experiment = False,
                                  inv_interval = 5
                                  ):
    assert(len(filename_dirs) == len(use_pads))
    
    base_dir_name = "model_to_evaluate"
    model_filename_base = "itr_{}.pkl"
    metric_name_ls = ['Rewards Mean', 'Rewards Std', 'Rewards Max', 'Rewards Min', 'Returns Mean', 'Returns Std', 'Returns Max', 'Returns Min', 'Actions Mean', 'Actions Std', 'Actions Max', 'Actions Min', 'Num Paths', 'Average Returns']
    varied_excel_filename_base = "scripts/{}/{}_comparison_stats_{}_{}_{}_{}.xlsx"
    stationary_excel_filename_base = "scripts/{}/{}_comparison_stats_fixed_{}.xlsx"

    result_dict = {}
    result_dict['policy_type'] = []
    result_dict['epoch'] = []
    result_dict['filename'] = []
    result_dict['use_pad'] = []
    result_dict['seed'] = []
    result_dict['rewards'] = []
    # for metric_name in metric_name_ls:
    #     result_dict[metric_name] = []

    # check if all file names exist first
    for dir_idx in range(len(filename_dirs)):
        dir_name = filename_dirs[dir_idx]
        use_pad = use_pads[dir_idx]
        for epoch in epochs:
            filename = os.path.join(dir_name, model_filename_base.format(epoch))
            filename_full = os.path.join(base_dir_name, filename)
            data = torch.load(filename_full)

    # perform experiments
    start_time = time.time()
    total_num_eval = len(filename_dirs) * len(epochs)
    cur_num_eval = 0
    for dir_idx in range(len(filename_dirs)):
        dir_name = filename_dirs[dir_idx]
        use_pad = use_pads[dir_idx]
        for epoch in epochs:
            filename = os.path.join(dir_name, model_filename_base.format(epoch))
            print('evaluating: {}, use_pad: {}, time: {:.2f}, progress: {}/{}'.format(filename, use_pad, time.time() - start_time, cur_num_eval, total_num_eval))
            filename_full = os.path.join(base_dir_name, filename)
            cur_num_eval += 1

            
            performance_stats = get_policy_average_returns_per_epoch(
                filename = filename_full, 
                eval_env = eval_env, 
                hasVariation = hasVariation,
                seeds = seeds, 
                use_gpu = use_gpu, 
                use_pad = use_pad,
                variation_attribute = variation_attribute,
                variation_type = variation_type,
                variation_amplitude = variation_amplitude,
                inv_interval=inv_interval,
                )

            for i in range(len(seeds)):
                result_dict['policy_type'].append(dir_name)
                result_dict['epoch'].append(epoch)
                result_dict['filename'].append(filename)
                result_dict['use_pad'].append(use_pad)
                result_dict['seed'].append(seeds[i])
                result_dict['rewards'].append(performance_stats[i])

            # result_dict['policy_type'].append(dir_name)
            # result_dict['epoch'].append(epoch)
            # result_dict['filename'].append(filename)
            # result_dict['use_pad'].append(use_pad)
            # for metric in metric_name_ls:
            #     result_dict[metric].append(performance_stats[metric])
    
    if hasVariation:
        excel_filename = varied_excel_filename_base.format(result_dir_name, datetime.now().strftime("%Y%m%d_%H%M"), eval_env,variation_attribute, variation_type, variation_amplitude)
    else:
        excel_filename = stationary_excel_filename_base.format(result_dir_name, datetime.now().strftime("%Y%m%d_%H%M"), eval_env)
    writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')

    # writing experiment result to file
    result_df = pd.DataFrame(result_dict)
    result_df.to_excel(writer, sheet_name="raw_data")

    # writing experiment parameters to file
    exp_setup = {}
    exp_setup['eval_env'] = [str(eval_env)]
    exp_setup['hasVariation'] = [str(hasVariation)]
    exp_setup['variation_attribute'] = [str(variation_attribute)]
    exp_setup['variation_type'] = [str(variation_type)]
    exp_setup['variation_amplitude'] = [str(variation_amplitude)]
    exp_setup['use_gpu'] = [str(use_gpu)]
    exp_setup['seeds'] = [str(seeds)]
    exp_setup['filename_dirs'] = [str(filename_dirs)]
    exp_setup['use_pads'] = [str(use_pads)]
    exp_setup['epochs'] = [str(epochs)]

    exp_setup_df = pd.DataFrame.from_dict(exp_setup, orient='index')
    exp_setup_df.to_excel(writer, sheet_name="exp_setup")

    writer.save()
    print('exported to file: {}'.format(excel_filename))

    # plot comparison graph
    plot_comparison(excel_filename, last_experiment = last_experiment)
    plot_single_epoch_comparison(excel_filename, 3000)
