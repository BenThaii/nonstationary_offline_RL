import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import time
import numpy as np

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

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220203-2220-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220203_2220_2022_02_03_22_20_14_0000--s-0")
# method_names.append("torch 1.4")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220123-1754-completed-torch1.4/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220123_1754_2022_01_23_17_54_16_0000--s-0")
# method_names.append("torch 1.4 revived")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220322-2032-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220322_2032_2022_03_22_20_32_28_0000--s-0")
# method_names.append("torch 1.4 new")

method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220323-2244-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220323_2244_2022_03_23_22_44_29_0000--s-0")
method_names.append("torch 1.4 full")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220320-1547-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220320_1547_2022_03_20_15_47_29_0000--s-0")
# method_names.append("antmaze-medium-diverse-v0 t1.4 noshared, qf:256-3, 256-1,[1:2], ss 40,000, mse invloss, interval 5")      # CQL + PAD, cql_alpha unstable, inv loss doesn't decrease

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220321-1759-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220321_1759_2022_03_21_17_59_41_0000--s-0")
# method_names.append("antmaze-medium-diverse-v0 t1.4 noshared, encoder relu, qf:256-3, 256-1,[1:1], ss 40,000, mse invloss, interval 1")      # CQL + PAD, cql_alpha stable

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220329-2342-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220329_2342_2022_03_29_23_42_01_0000--s-0")
# method_names.append("antmaze-medium-diverse-v0 t1.4 noshared, encoder relu, qf:256-3, 256-1,[1:2], ss 40,000, mse invloss, interval 1")      

method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220407-1544/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220407_1544_2022_04_07_15_44_48_0000--s-0")
method_names.append("antmaze-medium-diverse-v0 t1.4 noshared, encoder relu, qf:256-3, 256-1,[1:2], ss 40,000, mse invloss, interval 1 v2")      

method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220408-0840-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220408_0840_2022_04_08_08_40_28_0000--s-0")
method_names.append("antmaze-medium-diverse-v0 t1.4 noshared, encoder relu, qf:256-3, 256-1,[1:2], ss 40,000, mse invloss, interval 1 v2, relu")      

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220329-2343-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220329_2343_2022_03_29_23_43_20_0000--s-0")
# method_names.append("antmaze-medium-diverse-v0 t1.4 noshared, encoder relu, qf:256-3, 256-1,[1:1], ss 40,000, variational invloss, interval 1")      # CQL + PAD, cql_alpha stable

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220405-2345-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220405_2345_2022_04_05_23_45_43_0000--s-0")
# method_names.append("antmaze-medium-diverse-v0 t1.4 noshared, encoder relu, qf:256-3, 256-1,[1:2], ss 40,000, variational invloss, interval 1")      # CQL + PAD, cql_alpha stable


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220321-1330/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220321_1330_2022_03_21_13_30_35_0000--s-0")
# method_names.append("antmaze-medium-diverse-v0 t1.4 noshared, policy_lr=1e-4, encoder relu, qf:256-3, 256-1,[1:1], ss 40,000")      # quite stable








# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220130-0913-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220130_0913_2022_01_30_09_13_23_0000--s-0")
# method_names.append("torch 1.4 3 layers, 1st layer shared, no ss")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220130-0913-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220130_0913_2022_01_30_09_13_23_0000--s-0")
# method_names.append("torch 1.4 3 layers, 1st layer shared, no ss")

# EXPERIMENTING WITH LEARNING RATES (1e-4 is better)
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220130-1633-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220130_1633_2022_01_30_16_33_58_0000--s-0")
# method_names.append("torch 1.4 simple encoder & inv, with ss loss, lr=1e-4") # more convergent in many asspects than 1e-5, cql_alpha still saturates

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220130-2319-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220130_2319_2022_01_30_23_19_25_0000--s-0")
# method_names.append("torch 1.4 simple encoder & inv, with ss loss, lr=1e-5")

# EXPERIMENTING WITH WHEN TO START SELF-SUPERVISION (SS) (40,000 is better, delayed is better than no delay)
# Best SS model so far
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220131-1012-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220131_1012_2022_01_31_10_12_37_0000--s-0")
# method_names.append("torch 1.4 simple encoder & inv, lr=1e-5, ss 40,000")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220201-0825-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220201_0825_2022_02_01_08_25_00_0000--s-0")
# method_names.append("torch 1.4 simple encoder & inv, lr=1e-5, ss 100,000")

# EXPERIMENTING WITH SOFT UPDATES (update qf and encoder separately) (soft) (separate soft update between qf and encoder is bad (at the end), though is more consistent with the author of PAD)
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220201-1438-completed/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220201_1438_2022_02_01_14_38_22_0000--s-0")
# method_names.append("torch 1.4 simple encoder & inv, lr=1e-5, ss 40,000, soft sep")        # Good at the start, but bad at the end

# EXPERIMENTING WITH MORE LAYERS
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220202-1417-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220202_1417_2022_02_02_14_17_32_0000--s-0")
# method_names.append("torch 1.4 1h-layer encoder & inv, lr=1e-5, ss 40,000, soft sep")  # Worst Bad cql_alpha

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220202-1559-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220202_1559_2022_02_02_15_59_05_0000--s-0")
# method_names.append("torch 1.4 1h-layer encoder, simple inv, lr=1e-5, ss 40,000, soft sep")  #Bad cql_alpha

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220205-1801-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220205_1801_2022_02_05_18_01_22_0000--s-0")
# method_names.append("torch 1.4 128-4,[5:4], lr=1e-5, ss 40,000, soft sep")  

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220206-0000-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220206_0000_2022_02_06_00_00_50_0000--s-0")
# method_names.append("torch 1.4 128-0,[5:4], lr=1e-5, no ss, soft sep")  

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220206-0829/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220206_0829_2022_02_06_08_29_05_0000--s-0")
# method_names.append("torch 1.4 128-4,[1:1], lr=1e-5, no ss, soft sep")  

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220205-2304-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220205_2304_2022_02_05_23_04_42_0000--s-0")
# method_names.append("torch 1.4 128-1,[5:4], lr=1e-5, no ss, soft sep")  

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220205-1934-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220205_1934_2022_02_05_19_34_41_0000--s-0")
# method_names.append("torch 1.4 128-4,[5:4], lr=1e-5, no ss, soft sep")  

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220206-1258-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220206_1258_2022_02_06_12_58_03_0000--s-0")
# method_names.append("torch 1.4 256-0,[2:1], lr=1e-5, no ss, soft sep")  

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220206-2326-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220206_2326_2022_02_06_23_26_51_0000--s-0")
# method_names.append("torch 1.4 256-0,[2:1], lr=1e-5, no ss, true soft sep")



# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220202-1753-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220202_1753_2022_02_02_17_53_01_0000--s-0")
# method_names.append("torch 1.4 simple encoder, 1h-layer inv, lr=1e-5, ss 40,000, soft sep")  # Bad CQL alpha, but still better than the 2 above

# # EXPERIMENTING WITH 2 CQL ALPHAS
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/antmaze-medium-diverse-v0-20220203-1100-terminated/CQL_offline_mujoco_runs/antmaze-medium-diverse-v0_20220203_1100_2022_02_03_11_00_56_0000--s-0")
# method_names.append("torch 1.4 simple encoder, 1h-layer inv, lr=1e-5, ss 40,000, soft sep, 2cql_alpha")  

########################################HALF-CHEETAH
###### wrong cheetah (v0)
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-medium-v0-20220207-2230-completed/CQL_offline_mujoco_runs/halfcheetah-medium-v0_20220207_2230_2022_02_07_22_30_07_0000--s-0")
# method_names.append("cheetah-medium-v0 t1.4 standard")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-medium-v0-20220207-1841-terminated/CQL_offline_mujoco_runs/halfcheetah-medium-v0_20220207_1841_2022_02_07_18_41_00_0000--s-0")
# method_names.append("cheetah t1.4 256-0,[2:1], lr=1e-5, no ss, true soft sep")

# Cheetah V2
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220208-2246-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220208_2246_2022_02_08_22_46_22_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 standard")


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220209-1936-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220209_1936_2022_02_09_19_36_45_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-0,[2:1], lr=1e-5, no ss, true soft sep")

#######EXPERIMENTING WITH FREEZE LAYERS FOR INTERVALS OF UPDATES
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220210-2000-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220210_2000_2022_02_10_20_00_27_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-0,[2:1], lr=1e-5, no ss, true soft sep, interval 10:5")     # shown that freezing layers improve result

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220211-1902-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220211_1902_2022_02_11_19_02_48_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-0,[2:1], lr=1e-5, no ss, true soft sep, interval 5:2")      # same as 10:5

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220211-2043-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220211_2043_2022_02_11_20_43_22_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-0,[2:1], lr=1e-5, no ss, true soft sep, interval 15:7")      # worse than 10:5

#######EXPERIMENTING WITH NUMBER OF SHARED LAYERS (still no ss)
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220212-0917-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220212_0917_2022_02_12_09_17_50_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-1,[2:1], lr=1e-5, no ss, true soft sep, interval 10:5")         # still quite stable  

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220216-1023-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220216_1023_2022_02_16_10_23_57_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-1,[2:1], lr=1e-5, no ss, true soft sep, interval 5:2")         # better than 10:5, but avg does not improve

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220217-1007-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220217_1007_2022_02_17_10_07_43_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 1(128),[2:1](256), lr=1e-5, no ss, true soft sep, interval 5:2")         # smaller encoder layer is better than larger encoder layer, but avg return does not improve over time

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220212-1324-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220212_1324_2022_02_12_13_24_29_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-2,[2:1], lr=1e-5, no ss, true soft sep, interval 10:5")      # cql_alpha saturates
 
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220212-1714-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220212_1714_2022_02_12_17_14_36_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 128-2,[2:1], lr=1e-5, no ss, true soft sep, interval 10:5")      

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220212-2229-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220212_2229_2022_02_12_22_29_34_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 128-3,[3:1], lr=1e-5, no ss, true soft sep, interval 10:5")   # cql_alpha saturates @ 500 

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220213-0942-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220213_0942_2022_02_13_09_42_12_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 128-3,[3:1], lr=1e-5, no ss, true soft sep, interval 15:7")   # cql_alpha saturates @ 350 

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220213-1418-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220213_1418_2022_02_13_14_18_33_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 128-3,[3:1], lr=1e-5, no ss, true soft sep, interval 5:2")   # cql_alpha no longer saturates

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220213-0251-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220213_0251_2022_02_13_02_51_17_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 128-3,[2:1], lr=1e-5, no ss, true soft sep, interval 10:5")  # cql_alpha saturates @ 1000 

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220214-1453-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220214_1453_2022_02_14_14_53_28_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-2,[2:1], lr=1e-5, no ss, true soft sep, interval 5:2")      # alpha, (not cql_alpha) saturates @ 1800. cql_alpha is fine so far,  success rate of 50%

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220215-1952-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220215_1952_2022_02_15_19_52_43_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-2,[2:1], lr=1e-5, no ss, true soft sep, interval 5:2")      # repeat (no saturation)

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220215-2340-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220215_2340_2022_02_15_23_40_25_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-2,[2:1], lr=1e-5, no ss, true soft sep, interval 5:2")      # repeat 2 (with saturation -> 50% chance of saturation)


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220215-1017-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220215_1017_2022_02_15_10_17_20_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 256-2,[2:1], lr=1e-5, no ss, true soft sep, interval 4:2")      # cql_alpha saturates


# # ##### policy and q function does not have shared encoder
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220218-1005-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220218_1005_2022_02_18_10_05_13_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-1,[2:1], no ss")      # performance is similar to vanilla CQL

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220219-0958-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220219_0958_2022_02_19_09_58_45_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-2,[2:1], no ss")      # performance is slightly better than vanilla CQL

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220225-1513-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220225_1513_2022_02_25_15_13_54_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-1,[2:1], ss 40,000")      # more robust than vanilla CQL after 3000 epochs

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220305-1031-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220305_1031_2022_03_05_10_31_04_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-1,[2:1], ss 40,000, variational invloss")      # same as above -> dont need to reuse the policy

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220318-0818-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220318_0818_2022_03_18_08_18_25_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000, mse invloss")      # normal mse inv loss

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220317-0942-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220317_0942_2022_03_17_09_42_10_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000, variational invloss")      # normal variational inv loss

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220316-0933-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220316_0933_2022_03_16_09_33_25_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:2], ss 40,000, variational invloss")      # more layers in inverse network

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220319-0958-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220319_0958_2022_03_19_09_58_19_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:2], ss 40,000, mse invloss")      # more layers in inverse network

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220220-0958-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220220_0958_2022_02_20_09_58_22_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-2,[2:1], no ss, cont, BC 40,000")     

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220220-1031-terminated/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220220_1031_2022_02_20_10_31_53_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-2,[2:1], no ss, cont, no BC")      # see if no BC is better (not so much)

######## reusing the q-function from vanilla cql training
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220224-0813-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220224_0813_2022_02_24_08_13_10_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-2,[2:1], no ss, cont, no policy reuse")      # vanilla CQL train to obtain same number of training epoch as the ones with ss below (2 phases)


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220220-1121-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220220_1121_2022_02_20_11_21_31_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-2,[2:1], ss 40,000, cont, no BC, policy reuse")      # cql_alpha is stable, avg return is high

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/halfcheetah-expert-v2-20220221-2230-completed/CQL_offline_mujoco_runs/halfcheetah-expert-v2_20220221_2230_2022_02_21_22_30_05_0000--s-0")
# method_names.append("cheetah-expert-v2 t1.4 noshared, qf:256-3, 256-2,[2:1], ss 40,000, cont, no BC, no policy reuse")      # same as above -> dont need to reuse the policy


###### hopper environment
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220226-2216-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220226_2216_2022_02_26_22_16_50_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 standard, qf:256-3, 256-1,[1:1]")            # vanilla CQL, same number of layers as the original CQL in the paper  

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220228-0846-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220228_0846_2022_02_28_08_46_17_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], no ss")     # vanilla CQL using cql_noshared code, same number of layers as the original CQL in the paper  


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220227-1250-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220227_1250_2022_02_27_12_50_45_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000")    # CQL + PAD, same number of layers as above

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220326-1740-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220326_1740_2022_03_26_17_40_24_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000, mse invloss, true encoder")     

# # method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220313-1157-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220313_1157_2022_03_13_11_57_38_0000--s-0")
# # method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000, interval 5")    # same number of layers, with interval

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220327-2330-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220327_2330_2022_03_27_23_30_11_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:2], ss 40,000, mse invloss, true encoder")   


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220301-0018-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220301_0018_2022_03_01_00_18_05_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-2,[2:1], ss 40,000")    # CQL + PAD,  more layers

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220308-1834-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220308_1834_2022_03_08_18_34_30_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 standard, qf:256-3, 256-1,[2:_]")      # note: to have the same number of layers as the one below the one w/ inv loss


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220306-2259-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220306_2259_2022_03_06_22_59_26_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[2:1], ss 40,000, variational invloss")      # note: has more layers than cql + PAD w/ mse loss. Better than (1. vanilla CQL and 2. PAD w/ no ss and 3. cql+pad+mse loss)

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220309-2220-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220309_2220_2022_03_09_22_20_27_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[2:1], ss 40,000, mse invloss")      # note: has more layers than cql + PAD w/ mse loss. Better than (1. vanilla CQL and 2. PAD w/ no ss and 3. cql+pad+mse loss)


# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220307-2032-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220307_2032_2022_03_07_20_32_32_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000, variational invloss")      # better than 

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220326-1742-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220326_1742_2022_03_26_17_42_49_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000, variational invloss, true encoder")     

# # method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220310-2237-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220310_2237_2022_03_10_22_37_49_0000--s-0")
# # method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:2], ss 40,000, variational invloss")      # CQL + PAD, same number of layers (policy, q-functions, etc.), 1 extra inv network layer

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/hopper-expert-v2-20220327-2329-completed/CQL_offline_mujoco_runs/hopper-expert-v2_20220327_2329_2022_03_27_23_29_33_0000--s-0")
# method_names.append("hopper-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:2], ss 40,000, variational invloss, true encoder")   
 

########## WALKER2D environment
# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql_with_opal/d4rl/logger/CQL-offline-mujoco-runs/walker2d-expert-v2-20220324-2157-completed/CQL_offline_mujoco_runs/walker2d-expert-v2_20220324_2157_2022_03_24_21_57_00_0000--s-0")
# method_names.append("walker2d-expert-v2, torch 1.4")

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/walker2d-expert-v2-20220325-1201-completed/CQL_offline_mujoco_runs/walker2d-expert-v2_20220325_1201_2022_03_25_12_01_46_0000--s-0")
# method_names.append("walker2d-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000, mse invloss")    # CQL + PAD, same number of layers as above

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/walker2d-expert-v2-20220405-0009-completed/CQL_offline_mujoco_runs/walker2d-expert-v2_20220405_0009_2022_04_05_00_09_07_0000--s-0")
# method_names.append("walker2d-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:2], ss 40,000, mse invloss")    # CQL + PAD, same number of layers as abov

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/walker2d-expert-v2-20220402-2101-completed/CQL_offline_mujoco_runs/walker2d-expert-v2_20220402_2101_2022_04_02_21_01_08_0000--s-0")
# method_names.append("walker2d-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:1], ss 40,000, variational invloss")    # CQL + PAD, same number of layers as above

# method_dirs.append("/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/logger/CQL-offline-mujoco-runs/walker2d-expert-v2-20220404-0034-completed/CQL_offline_mujoco_runs/walker2d-expert-v2_20220404_0034_2022_04_04_00_34_40_0000--s-0")
# method_names.append("walker2d-expert-v2 t1.4 noshared, qf:256-3, 256-1,[1:2], ss 40,000, variational invloss")    # CQL + PAD, same number of layers as abov


assert(len(method_dirs) == len(method_names))
progress_file_name = "progress.csv"


num_metrics = len(comparison_metrics)
num_dim1 = math.floor(math.sqrt(num_metrics))
num_dim2 = math.ceil(num_metrics/num_dim1)

min_idx = 1
# min_idx = 1250
max_idx = 1400

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
            # if metric == "trainer/min_q1_loss" or metric == "trainer/min_q2_loss":
            if False:
                if plt_method == "normal":
                    subplot.plot(method_progress["Epoch"][1:], np.divide(list(method_progress[metric][1:]),list(method_progress["trainer/Alpha_prime"][:-1])), label = method_names[method_idx] + "_pre_cql_alpha", alpha=0.5)
                elif plt_method == "min":
                    subplot.plot(method_progress["Epoch"][min_idx:max_idx], np.divide(list(method_progress[metric][min_idx:max_idx]),list(method_progress["trainer/Alpha_prime"][min_idx-1:max_idx-1])), label = method_names[method_idx] + "_pre_cql_alpha", alpha=0.5)
                
                
            elif metric in method_progress:
                print(metric)
                if plt_method == "normal":
                    subplot.plot(method_progress["Epoch"], method_progress[metric], label = method_names[method_idx], alpha=0.5)
                elif plt_method == "min":
                    subplot.plot(method_progress["Epoch"][min_idx:max_idx], method_progress[metric][min_idx:max_idx], label = method_names[method_idx], alpha=0.5)
                print("{}: {}/{}".format(method_names[method_idx], method_progress[metric].sum(), method_progress[metric].count()))
                break
    subplot.set_title(metric_vals[0])
    subplot.legend(prop={'size': 6})
plt.show()
time.sleep(15)


    


        

# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()