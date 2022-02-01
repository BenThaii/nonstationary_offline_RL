import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import time

comparison_metrics = []
comparison_metrics.append(["trainer/QF1 Loss"])
comparison_metrics.append(["trainer/QF2 Loss"])
comparison_metrics.append(["trainer/Policy Loss"])
comparison_metrics.append(["trainer/Alpha Loss"])
comparison_metrics.append(["trainer/cql_alpha_loss", "trainer/alpha prime loss"])
comparison_metrics.append(["trainer/Alpha"])
comparison_metrics.append(["trainer/cql_alpha", "trainer/Alpha_prime"])
comparison_metrics.append(["trainer/Policy mu Std"])
comparison_metrics.append(["trainer/Log Pis Mean"])
comparison_metrics.append(["trainer/Log Pis Std"])
comparison_metrics.append(["trainer/self-supervised inv loss"])
# comparison_metrics.append(["evaluation/path length Mean"])
comparison_metrics.append(["evaluation/Average Returns"])
comparison_metrics.append(["trainer/min_qf1_loss_tau"])
comparison_metrics.append(["trainer/min_qf2_loss_tau"])
comparison_metrics.append(["trainer/min_q1_loss"])
comparison_metrics.append(["trainer/min_q2_loss"])
comparison_metrics.append(["trainer/Q1 Predictions Mean", "trainer/Q1 Predictions_need_to_update_eval_statistics Mean"])
comparison_metrics.append(["trainer/Q2 Predictions Mean"])
comparison_metrics.append(["trainer/cql_alpha_grad"])
comparison_metrics.append(["exploration/Average Returns"])
# comparison_metrics.append(["evaluation/Actions Mean"])



method_dirs = []
method_names = []


method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220130-0913-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220130_0913_2022_01_30_09_13_23_0000--s-0")
method_names.append("torch 1.4 3 layers, 1st layer shared, no ss")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220130-1633-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220130_1633_2022_01_30_16_33_58_0000--s-0")
# method_names.append("torch 1.4  simple encoder & inv, with ss loss, lr=1e-4")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220130-2319-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220130_2319_2022_01_30_23_19_25_0000--s-0")
# method_names.append("torch 1.4  simple encoder & inv, with ss loss, lr=1e-5")

method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220131-1012-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220131_1012_2022_01_31_10_12_37_0000--s-0")
method_names.append("torch 1.4  simple encoder & inv, lr=1e-5, ss start 40,000")

method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220201-0825/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220201_0825_2022_02_01_08_25_00_0000--s-0")
method_names.append("torch 1.4  simple encoder & inv, lr=1e-5, ss start 100,000")


assert(len(method_dirs) == len(method_names))
progress_file_name = "progress.csv"


num_metrics = len(comparison_metrics)
num_dim1 = math.floor(math.sqrt(num_metrics))
num_dim2 = math.ceil(num_metrics/num_dim1)



min_idx = 0
# max_idx = 10
# max_idx = 300
max_idx = 500

# plt_method = "min"
plt_method = "normal"
for method in method_dirs:
    method_progress = pd.read_csv(os.path.join(method, progress_file_name))
    max_idx = min(len(method_progress), max_idx)


fig,axs = plt.subplots(num_dim1, num_dim2, figsize=(30, 15))
for metric_idx, metric_vals in enumerate(comparison_metrics):
    row = math.floor((metric_idx) / num_dim2)
    col = (metric_idx)  % num_dim2

    if num_dim1 == 1 and num_dim2 == 1:
        subplot = axs
    else:
        subplot = axs[row,col]
    for method_idx in range(len(method_dirs)):
        method_progress = pd.read_csv(os.path.join(method_dirs[method_idx], progress_file_name))
        # plt.scatter(method_progress["Epoch"], method_progress[comparison_metric])
        for metric in metric_vals:
            if metric in method_progress:
                if plt_method == "normal":
                    subplot.plot(method_progress["Epoch"], method_progress[metric], label = method_names[method_idx], alpha=0.5)
                elif plt_method == "min":
                    subplot.plot(method_progress["Epoch"][:max_idx], method_progress[metric][:max_idx], label = method_names[method_idx], alpha=0.5)
                print("{}: {}/{}".format(method_names[method_idx], method_progress[metric].sum(), method_progress[metric].count()))
                break
    subplot.set_title(metric_vals[0])
    subplot.legend(prop={'size': 6})
plt.show()
time.sleep(5)


    


        

# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()