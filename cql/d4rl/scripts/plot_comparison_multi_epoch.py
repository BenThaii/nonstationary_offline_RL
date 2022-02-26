import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comparison_helper_pad import plot_comparison


# excel_filename = "scripts/result/20220225_1352_comparison_stats_gravity_linear-increase_20.xlsx"
excel_filename = "scripts/20220225_1939_comparison_stats_gravity_linear-increase_5.xlsx"
excel_filename = "scripts/20220225_2044_comparison_stats_gravity_linear-increase_10.xlsx"
excel_filename = "scripts/20220225_2055_comparison_stats_gravity_linear-increase_15.xlsx"
excel_filename = "scripts/20220225_2216_comparison_stats_fixed_modifiedcheetah.xlsx"
excel_filename = "scripts/20220225_2312_comparison_stats_HalfCheetahModified-v2_gravity_linear-increase_5.xlsx"
excel_filename = "scripts/20220225_2329_comparison_stats_HalfCheetahModified-v2_gravity_linear-increase_10.xlsx"
excel_filename = "scripts/20220225_2358_comparison_stats_HalfCheetahModified-v2_dof_friction_linear-increase_5.xlsx"
excel_filename = "scripts/20220226_0029_comparison_stats_HalfCheetah-v2_dof_friction_linear-increase_5.xlsx"
excel_filename = "scripts/20220226_0933_comparison_stats_HalfCheetah-v2_dof_friction_linear-increase_10.xlsx"


plot_comparison(excel_filename)