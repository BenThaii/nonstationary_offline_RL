import pandas as pd
import os
import matplotlib.pyplot as plt

comparison_metric = "evaluation/Returns Mean"
# comparison_metric = "evaluation/path length Mean"

# comparison_metric = "evaluation/Average Returns"
# comparison_metric = "evaluation/Actions Mean"

method_dir = []
method_name = []





# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/walker2d-random-v0-myfix-no-detach-no-retain-graph/CQL_offline_mujoco_runs/walker2d-random-v0_20211115_0805_2021_11_15_08_05_30_0000--s-0")
# method_name.append("my fix_old")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/walker2d-random-v0-20211115-1233/CQL_offline_mujoco_runs/walker2d-random-v0_20211115_1233_2021_11_15_12_33_44_0000--s-0")
# method_name.append("torch 1.4")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/walker2d-random-v0-contributor-20211113-18-34-42/CQL_offline_mujoco_runs/walker2d-random-v0_20211113_18_34_42_2021_11_13_18_34_42_0000--s-0")
# method_name.append("contributor fix")

# method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/walker2d-random-v0-20211116-1121/CQL_offline_mujoco_runs/walker2d-random-v0_20211116_1121_2021_11_16_11_21_47_0000--s-0")
# method_name.append("my fix")



method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211123-2316/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211123_2316_2021_11_23_23_16_02_0000--s-0")
method_name.append("antmaze myfix")

method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-original-torch14-20211121-1801/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211121_1801_2021_11_21_18_01_11_0000--s-0")
method_name.append("antmaze torch 1.4")

method_dir.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-play-v0-20211124-2140/CQL_offline_mujoco_runs/antmaze-medium-play-v0_20211124_2140_2021_11_24_21_40_42_0000--s-0")
method_name.append("antmaze myfix new lagrange")


assert(len(method_dir) == len(method_name))


progress_file_name = "progress.csv"

for idx in range(len(method_dir)):
    method_progress = pd.read_csv(os.path.join(method_dir[idx], progress_file_name))
    plt.plot(method_progress[comparison_metric])


plt.legend(method_name)
plt.show()