from comparison_helper_pad import *

# import necessary libraries
import os
import glob
  

folder_paths = []
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220226_1751_cheetah_normal_and_modified')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220226_1958_cheetah_broken')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220228_0826_hopper_normal_pad')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220304_2316_hopper_normal_padnormal_pad')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220306_1843_cheetah_normal_and_modified_variational')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220307_1835_hopper_normal_morelayerpad_morelayervariationalpad')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220308_1427_hopper_normal_msepad_variationalpad')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220309_1009_hopper_noramlmorelayer_padmorelayer')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220310_1701_hopper_normalmorelayer_padmorelayermse_padmorelayervariational')
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220311_1902_hopper_normal_padmse_padvariational_padvariationalinvextralayer')

# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220226_1751_cheetah_normal_and_modified_mse')

# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220226_1958_cheetah_broken_mse') # not detailed rewards
# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220315_1746_cheetah_msepolicyextralayer_variationalpolicyextralayer_interval5')

folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220320_0815_cheetah_mse_variational_mseinvextralayer_variationalinvextralayer')
folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220327_1657_cheetahbroken_mse_variational_mseinvextralayer_variationalinvextralayer')
folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220327_2300_hopper_mserelu_variationalrelu_mseinvextralayer_variationalinvextralayer')
folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220405_1826_walker2d_mse_variational_mseinvextralayer_variationalinvextralayer')


# folder_paths.append('/home/ben/offline_RL/nonstationary_offline_RL/cql/d4rl/scripts/20220409_1315_antmaze_mse_variational_mseinvextralayer_variationalinvextralayer')

for folder_path in folder_paths:
    # use glob to get all the csv files
    # in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    
    # loop over the list of csv files
    for f_idx in range(len(csv_files)):
        # plot_comparison(csv_files[f_idx], mean_only=True)
        # plot_single_epoch_comparison(csv_files[f_idx], 3000, mean_only=True)

        plot_comparison(csv_files[f_idx])
        plot_single_epoch_comparison(csv_files[f_idx], 3000)

