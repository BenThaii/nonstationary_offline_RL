import pandas as pd
import os
import matplotlib.pyplot as plt
import math

comparison_metrics = []
comparison_metrics.append("trainer/QF1 Loss")
comparison_metrics.append("trainer/QF2 Loss")
comparison_metrics.append("trainer/Policy Loss")
# comparison_metrics.append("evaluation/path length Mean")
comparison_metrics.append("evaluation/Average Returns")
comparison_metrics.append("evaluation/Actions Mean")

method_dir = []
method_name = []




method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-play-v0-myfix-lagrange-5-20211124-2140/CQL_offline_mujoco_runs/antmaze-medium-play-v0_20211124_2140_2021_11_24_21_40_42_0000--s-0")
method_name.append("antmaze myfix new lagrange")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-myfix-lagrange-neg1-20211123-2316/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211123_2316_2021_11_23_23_16_02_0000--s-0")
# method_name.append("antmaze myfix")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-original-torch14-lagrange-neg1-20211121-1801/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211121_1801_2021_11_21_18_01_11_0000--s-0")
# method_name.append("antmaze torch 1.4")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211130-2340-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211130_2340_2021_11_30_23_40_14_0000--s-0")
# method_name.append("antmaze opal no rewards")




method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211202-2323/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211202_2323_2021_12_02_23_23_51_0000--s-0")
method_name.append("antmaze opal 162 rewards")

method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211207-1111-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211207_1111_2021_12_07_11_11_11_0000--s-0")
method_name.append("antmaze opal 1845 rewards")

method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211210-1211-interrupted/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211210_1211_2021_12_10_12_11_22_0000--s-0")
method_name.append("antmaze opal 3576 rewards")

method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211216-1458-interrupted/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211216_1458_2021_12_16_14_58_09_0000--s-0")
method_name.append("antmaze opal 3576 rewards_new-primitive-new-terminals")




# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211218-1205/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211218_1205_2021_12_18_12_05_55_0000--s-0")
# method_name.append("opal 3576 r- cql_alpha=1.0")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211218-1253/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211218_1253_2021_12_18_12_53_44_0000--s-0")
# method_name.append("opal 3576 r- cql_alpha=5.0-training")


assert(len(method_dir) == len(method_name))
progress_file_name = "progress.csv"


num_metrics = len(comparison_metrics)
num_dim1 = math.floor(math.sqrt(num_metrics))
num_dim2 = math.ceil(math.sqrt(num_metrics))
fig,axs = plt.subplots(num_dim1, num_dim2)

for metric_idx, metric in enumerate(comparison_metrics):
    
    row = math.floor((metric_idx) / num_dim2)
    col = (metric_idx)  % num_dim2
    print(row, col)

    if num_dim1 == 1 and num_dim2 == 1:
        subplot = axs
    else:
        subplot = axs[row,col]
    for method_idx in range(len(method_dir)):
        method_progress = pd.read_csv(os.path.join(method_dir[method_idx], progress_file_name))
        # plt.scatter(method_progress["Epoch"], method_progress[comparison_metric])
        subplot.plot(method_progress["Epoch"], method_progress[metric])
        subplot.set_title(metric)
        subplot.legend(method_name, prop={'size': 6})

        print("{}: {}/{}".format(method_name[method_idx], method_progress[metric].sum(), method_progress[metric].count()))

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()