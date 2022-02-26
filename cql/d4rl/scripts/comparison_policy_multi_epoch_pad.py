
from comparison_helper_pad import compare_policies_multi_epochs
import os
from datetime import datetime





filename_dirs = ['vanilla_3000', 'pad_3000', 'pad_3000']
use_pads = [False, False, True]
epochs = range(2100, 3001, 300)
seeds = range(0,1000, 100)

eval_env_ls = []
hasVariation_ls = []
variation_attribute_ls = []
variation_type_ls = []
variation_amplitude_ls = []


#experiment 1: no change
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(False)
variation_attribute_ls.append(None)
variation_type_ls.append(None)
variation_amplitude_ls.append(None)

#experiment 2: original env, gravity linearly increase by 5
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('gravity')
variation_type_ls.append('linear-increase')
variation_amplitude_ls.append(5)


#experiment 3: original env, gravity linearly increase by 10
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('gravity')
variation_type_ls.append('linear-increase')
variation_amplitude_ls.append(10)

#experiment 4: original env, gravity linearly decrease by 5
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('gravity')
variation_type_ls.append('linear-decrease')
variation_amplitude_ls.append(5)

#experiment 5: original env, gravity linearly decrease by 10
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('gravity')
variation_type_ls.append('linear-decrease')
variation_amplitude_ls.append(10)


#experiment 6: modified Cheetah env, no variation
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(False)
variation_attribute_ls.append(None)
variation_type_ls.append(None)
variation_amplitude_ls.append(None)

#experiment 7: modified Cheetah env, gravity linearly increase by 5
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('gravity')
variation_type_ls.append('linear-increase')
variation_amplitude_ls.append(5)


#experiment 8: modified Cheetah env, gravity linearly increase by 10
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('gravity')
variation_type_ls.append('linear-increase')
variation_amplitude_ls.append(10)

#experiment 9: modified Cheetah env, gravity linearly decrease by 5
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('gravity')
variation_type_ls.append('linear-decrease')
variation_amplitude_ls.append(5)

#experiment 10: modified Cheetah env, gravity linearly decrease by 10
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('gravity')
variation_type_ls.append('linear-decrease')
variation_amplitude_ls.append(10)

#experiment 11: original env, dof_friction linearly increase by 5
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('dof_friction')
variation_type_ls.append('linear-increase')
variation_amplitude_ls.append(5)


#experiment 12: original env, dof_friction linearly increase by 10
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('dof_friction')
variation_type_ls.append('linear-increase')
variation_amplitude_ls.append(10)

#experiment 13: original env, dof_friction linearly decrease by 5
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('dof_friction')
variation_type_ls.append('linear-decrease')
variation_amplitude_ls.append(5)

#experiment 14: original env, dof_friction linearly decrease by 10
eval_env_ls.append('HalfCheetah-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('dof_friction')
variation_type_ls.append('linear-decrease')
variation_amplitude_ls.append(10)

#experiment 15: modified Cheetah env, dof_friction linearly increase by 5
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('dof_friction')
variation_type_ls.append('linear-increase')
variation_amplitude_ls.append(5)


#experiment 16: modified Cheetah env, dof_friction linearly increase by 10
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('dof_friction')
variation_type_ls.append('linear-increase')
variation_amplitude_ls.append(10)

#experiment 17: modified Cheetah env, dof_friction linearly decrease by 5
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('dof_friction')
variation_type_ls.append('linear-decrease')
variation_amplitude_ls.append(5)

#experiment 18: modified Cheetah env, dof_friction linearly decrease by 10
eval_env_ls.append('HalfCheetahModified-v2')
hasVariation_ls.append(True)
variation_attribute_ls.append('dof_friction')
variation_type_ls.append('linear-decrease')
variation_amplitude_ls.append(10)





dir_name = datetime.now().strftime("%Y%m%d_%H%M")
os.mkdir("scripts/{}".format(dir_name))
for index in range(len(eval_env_ls)):
    last_experiment = index == len(eval_env_ls) - 1
    compare_policies_multi_epochs(
        eval_env= eval_env_ls[index],
        hasVariation= hasVariation_ls[index],
        variation_attribute= variation_attribute_ls[index],
        variation_type= variation_type_ls[index],
        variation_amplitude= variation_amplitude_ls[index],
        filename_dirs= filename_dirs,
        use_pads= use_pads,
        epochs= epochs,
        seeds = seeds,
        result_dir_name = dir_name,
        last_experiment= last_experiment
    )
    print("evaluated experiment {}/{}".format(index + 1, len(eval_env_ls)))