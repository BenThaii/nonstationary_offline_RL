import pandas as pd
import os
import matplotlib.pyplot as plt

# comparison_metric = "evaluation/Returns Mean"
# comparison_metric = "evaluation/path length Mean"

comparison_metric = "evaluation/Average Returns"
# comparison_metric = "evaluation/Actions Mean"

method_dir = []
method_name = []





# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-myfix-lagrange-neg1-20211123-2316/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211123_2316_2021_11_23_23_16_02_0000--s-0")
# method_name.append("antmaze myfix")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-original-torch14-lagrange-neg1-20211121-1801/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211121_1801_2021_11_21_18_01_11_0000--s-0")
# method_name.append("antmaze torch 1.4")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-play-v0-myfix-lagrange-5-20211124-2140/CQL_offline_mujoco_runs/antmaze-medium-play-v0_20211124_2140_2021_11_24_21_40_42_0000--s-0")
# method_name.append("antmaze myfix new lagrange")



# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211130-2340-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211130_2340_2021_11_30_23_40_14_0000--s-0")
# method_name.append("antmaze opal no rewards")


# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211202-2323/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211202_2323_2021_12_02_23_23_51_0000--s-0")
# method_name.append("antmaze opal 162 rewards")

method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211207-1111-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211207_1111_2021_12_07_11_11_11_0000--s-0")
method_name.append("antmaze opal 1845 rewards")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211208-2347-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211208_2347_2021_12_08_23_47_21_0000--s-0")
# method_name.append("antmaze opal 1845 rewards-continued-with BC")

method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211210-1211-interrupted/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211210_1211_2021_12_10_12_11_22_0000--s-0")
method_name.append("antmaze opal 3576 rewards")

method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211214-0048/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211214_0048_2021_12_14_00_48_18_0000--s-0")
method_name.append("antmaze opal 3576 rewards_new-primitive")


assert(len(method_dir) == len(method_name))


progress_file_name = "progress.csv"

for idx in range(len(method_dir)):
    method_progress = pd.read_csv(os.path.join(method_dir[idx], progress_file_name))
    # plt.scatter(method_progress["Epoch"], method_progress[comparison_metric])
    plt.plot(method_progress["Epoch"], method_progress[comparison_metric])
    print("{}: {}/{}".format(method_name[idx], method_progress[comparison_metric].sum(), method_progress[comparison_metric].count()))
    


plt.legend(method_name)
plt.show()