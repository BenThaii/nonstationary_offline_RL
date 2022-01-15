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
# comparison_metrics.append(["evaluation/path length Mean"])
# comparison_metrics.append(["evaluation/Average Returns"])
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




# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-play-v0-myfix-lagrange-5-20211124-2140/CQL_offline_mujoco_runs/antmaze-medium-play-v0_20211124_2140_2021_11_24_21_40_42_0000--s-0")
# method_names.append("antmaze myfix new lagrange = 5")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-myfix-lagrange-neg1-20211123-2316/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211123_2316_2021_11_23_23_16_02_0000--s-0")
# method_names.append("antmaze myfix")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-original-torch14-lagrange-neg1-20211121-1801/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211121_1801_2021_11_21_18_01_11_0000--s-0")
# method_names.append("antmaze torch 1.4")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211130-2340-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211130_2340_2021_11_30_23_40_14_0000--s-0")
# method_names.append("antmaze opal no rewards")




# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211207-1111-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211207_1111_2021_12_07_11_11_11_0000--s-0")
# method_names.append("antmaze opal 1845 rewards")
#########################Simplest

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211210-1211-interrupted/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211210_1211_2021_12_10_12_11_22_0000--s-0")
# method_names.append("antmaze opal 3576 rewards")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211216-1458-interrupted/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211216_1458_2021_12_16_14_58_09_0000--s-0")
# method_names.append("antmaze opal 3576 rewards_new-primitive-new-terminals")



# # testing only

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211218-1205/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211218_1205_2021_12_18_12_05_55_0000--s-0")
# method_names.append("opal 3576 r- cql_alpha=1.0")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211218-1253/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211218_1253_2021_12_18_12_53_44_0000--s-0")
# method_names.append("opal 3576 r- cql_alpha=5.0")




# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211218-1356/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211218_1356_2021_12_18_13_56_26_0000--s-0")
# method_names.append("opal 1845r, new primitive, cql_alpha=1.0, discounted, no pretrain")




# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211218-1539/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211218_1539_2021_12_18_15_39_09_0000--s-0")
# method_names.append("opal 1845r, default params ")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211218-2209/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211218_2209_2021_12_18_22_09_59_0000--s-0")
# method_names.append("opal 1845r, new primitive, cql_alpha=1.0, discounted, without nll only, w/pretrain")






# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211219-0847-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211219_0847_2021_12_19_08_47_55_0000--s-0")
# method_names.append("opal 1845r, new primitive, cql_alpha=1.0, discounted, with nll only, w/ pretrain")
# #########################Best

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211220-1859-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211220_1859_2021_12_20_18_59_01_0000--s-0")
# method_names.append("opal 1845r, new primitive, suggest params")
# ########################Most Promising, loss decrease slower rate

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211222-1011/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211222_1011_2021_12_22_10_11_53_0000--s-0")
# method_names.append("c=1, no decoder/encoder cql_opal_test, cql_antmax_new_opal_test")
####shown that core opal works

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211223-1220-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211223_1220_2021_12_23_12_20_31_0000--s-0")
# method_names.append("opal 1845r, cont.train.prim.150, suggested params, wrong dset")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211224-0900-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211224_0900_2021_12_24_09_00_56_0000--s-0")
# method_names.append("opal 1845r, cont.train.prim.150, suggested params, no shuffle")
####trying to show that the bug is due to bad primitive

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211226-0945-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211226_0945_2021_12_26_09_45_06_0000--s-0")
# method_names.append("opal 1845r, cont.train.prim.150, suggested params, shuffled")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211228-1210-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211228_1210_2021_12_28_12_10_51_0000--s-0")
# method_names.append("opal 1845r, cont.train.prim.150, suggested params, shuffled, kld robust")
####trying to show that the bug is due to bad primitive

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20211230-1330-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20211230_1330_2021_12_30_13_30_42_0000--s-0")
# method_names.append("opal 1845r, cont.train.prim.150, suggested params, shuffled, kld robust, max q")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220102-1247-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220102_1247_2022_01_02_12_47_09_0000--s-0")
# method_names.append("opal 1845r, author utils, old cql data approach")
# ###trying to show that the bug is due to bad primitive

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220103-1234-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220103_1234_2022_01_03_12_34_46_0000--s-0")
# method_names.append("opal 1845r, author utils, new cql data approach")
###trying to show that the bug is due to bad primitive


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220103-1803-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220103_1803_2022_01_03_18_03_32_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220105-2207-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220105_2207_2022_01_05_22_07_11_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, no alphas, check df decreasing")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220106-1131-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220106_1131_2022_01_06_11_31_22_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, min_q_weight, small policy lr")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220106-1358-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220106_1358_2022_01_06_13_58_54_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, small policy lr")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220106-2217-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220106_2217_2022_01_06_22_17_32_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, only qf loss")
# shown that qf loss can decrease, though they get very unstable at the end

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220107-0914-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220107_0914_2022_01_07_09_14_40_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, qf + policy loss only")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220107-1439-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220107_1439_2022_01_07_14_39_20_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, qf + policy + cql_alpha loss")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220109-2223-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220109_2223_2022_01_09_22_23_20_0000--s-0")
# method_names.append("opal 1845r, author utils+cql, unshuffled, torch 1.4, qf + policy + alpha loss")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220107-1936-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220107_1936_2022_01_07_19_36_51_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, qf + policy +cql_alpha loss, cql_alpha = 5")


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220107-2145-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220107_2145_2022_01_07_21_45_52_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, all loss except latent, cql_alpha = 5")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220107-2239-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220107_2239_2022_01_07_22_39_10_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, all loss except latent, cql_alpha=5, cql_tau=-1 (like CQL)")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220108-1253-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220108_1253_2022_01_08_12_53_22_0000--s-0")
# method_names.append("opal 1845r, author utils, cql, torch 1.4, all loss except latent, cql_alpha=5, cql_tau=-1 (like CQL)_unshuffled_monitor_cqlalpha")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220108-1636-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220108_1636_2022_01_08_16_36_26_0000--s-0")
# method_names.append("cql_monitor_cqlalpha, lagrange = 5")


method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220109-1139-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220109_1139_2022_01_09_11_39_18_0000--s-0")
method_names.append("cql_opal, unshuffled, monitor_cqlalpha, lagrange = 5")


method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220110-1052-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220110_1052_2022_01_10_10_52_53_0000--s-0")
method_names.append("cql_opal, unshuffled, all loss, lagrange = 5, max_length 2000")

assert(len(method_dirs) == len(method_names))
progress_file_name = "progress.csv"


num_metrics = len(comparison_metrics)
num_dim1 = math.floor(math.sqrt(num_metrics))
num_dim2 = math.ceil(num_metrics/num_dim1)



min_idx = 0
# max_idx = 10
# max_idx = 300
max_idx = 500

plt_method = "min"
# plt_method = "normal"
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
