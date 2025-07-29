import os
os.environ['TIMELOOP_BASE_PATH'] = "/home/askiad/timeloop"
if "TIMELOOP_BASE_PATH" not in os.environ:
    timeloop_path = input("Please specify the path to Timeloop repo (default: " +  os.getcwd() + "/../):" ) or os.getcwd() + "/../"
    os.environ["TIMELOOP_BASE_PATH"] = timeloop_path
os.environ["TIMELOOP_ENABLE_FIRST_READ_ELISION"] = "1"
print('Path to timeloop repo: ', os.environ["TIMELOOP_BASE_PATH"])
import pathlib
import pandas as pd
import numpy as np
import src.utils as utils
import src.plots as plots
import matplotlib.pyplot as plt

workload_dir = pathlib.Path('./workloads')
config_dir = pathlib.Path('./configs/multi-einsum')
output_dir = pathlib.Path('./outputs/multi-einsum')

fig_dir = pathlib.Path('./figs')
fig_dir.mkdir(parents=True, exist_ok=True)

model_name = 'gpt3-6.7b'
batch_size=16
num_heads=32
input_format='chain'

# Get workload chain specs
chains, layers_dict, _ = utils.get_chain_config(workload_dir, output_dir, model_name=model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size)
for chain_idx, chain in enumerate(chains):
    print(f'chain {chain_idx}: {chain}')
# Calculate theoretical optimal bounds
optimal_accesses, optimal_accesses_fused = plots.get_optimal_performance(chains, layers_dict, num_heads)

# FFN is the 4th chain in the gpt3-6.7b workload
chain_idx = 4

# Get path name to different bounds
csv_unfused = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size,
    constraint_config='', scheme='opt',  enable_fusion=False, matmul_only=False)
csv_tiled_fusion = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size
    )
csv_tiled_segmented_fusion = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size,
    eval_slices=True)
csv_tiled_fusion_flash = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size,
    constraint_config='_relax_io_kn_flash')
csv_tiled_segmented_fusion_flash = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size,
    constraint_config='_relax_io_kn_flash', eval_slices=True)

# Load bounds to pandas dataframe
df_unfused = pd.read_csv(csv_unfused)
df_tiled_fusion =  pd.read_csv(csv_tiled_fusion)
df_tiled_segmented_fusion = pd.read_csv(csv_tiled_segmented_fusion)
df_tiled_fusion_flash = pd.read_csv(csv_tiled_fusion_flash)
df_tiled_segmented_fusion_flash = pd.read_csv(csv_tiled_segmented_fusion_flash)

ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx],
                                    df_nochain=(df_unfused, 'No Fusion'),
                                    df_slice=(df_tiled_segmented_fusion, 'Segmented Tiled Fusion'),
                                    df_relax_io=(df_tiled_fusion_flash, 'Tiled Fusion Flash'),
                                    df_relax_io_kn=(df_tiled_segmented_fusion_flash, 'Segmented Tiled Fusion Flash'),
                                    xbound=(10**4, None), ybound=(None, 10**12), logx=True, logy=True,
                                    figsize=(3, 2.5), max_effect_size=True,
                                    plot_cache=['100MB'])
legend = ax.legend(loc='upper center', bbox_to_anchor=(1.5, 1), fontsize=8)
ax.set_title("4th chain (FFN) in gpt3-6.7b")
plt.savefig(f"{fig_dir}/chainFFN.svg", format="svg", bbox_inches="tight")


model_name = 'gpt3-6.7b'
batch_size=16
num_heads=32
input_format='chain'
arch_prefix=''

# Get workload chain specs
chains, layers_dict, _ = utils.get_chain_config(workload_dir, output_dir, model_name=model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size)
for chain_idx, chain in enumerate(chains):
    print(f'chain {chain_idx}: {chain}')
# Calculate theoretical optimal bounds
optimal_accesses, optimal_accesses_fused = plots.get_optimal_performance(chains, layers_dict, num_heads, batch_size)

# 6 GEMMs is the 0th chain in the gpt3-6.7b workload
chain_idx = 0

# Get path name to different bounds
csv_unfused = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size, arch_prefix=arch_prefix,
    constraint_config='', scheme='opt', enable_fusion=False, matmul_only=False)
csv_tiled_fusion = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size, arch_prefix=arch_prefix
    )
csv_tiled_segmented_fusion = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size, arch_prefix=arch_prefix,
    eval_slices=True)
csv_tiled_fusion_flash = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size, arch_prefix=arch_prefix,
    constraint_config='_relax_io_kn_flash')
csv_tiled_segmented_fusion_flash = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size, arch_prefix=arch_prefix, 
    constraint_config='_relax_io_kn_flash', eval_slices=True)

df_unfused = pd.read_csv(csv_unfused)
df_tiled_fusion =  pd.read_csv(csv_tiled_fusion)
df_tiled_segmented_fusion = pd.read_csv(csv_tiled_segmented_fusion)
df_tiled_fusion_flash = pd.read_csv(csv_tiled_fusion_flash)
df_tiled_segmented_fusion_flash = pd.read_csv(csv_tiled_segmented_fusion_flash)

ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx],
                                    df_nochain=(df_unfused, 'No Fusion'),
                                    df_slice=(df_tiled_segmented_fusion, 'Segmented Tiled Fusion'),
                                    df_relax_io=(df_tiled_fusion_flash, 'Tiled Fusion Flash'),
                                    df_relax_io_kn=(df_tiled_segmented_fusion_flash, 'Segmented Tiled Fusion Flash'),
                                    xbound=(10**4, None), ybound=(None, 10**12), logx=True, logy=True,
                                    figsize=(3, 2.5), max_effect_size=True,
                                    plot_cache=['100MB'])
legend = ax.legend(loc='upper center', bbox_to_anchor=(1.5, 1), fontsize=8)
ax.set_title("0th chain (6 GEMMS) in gpt3-6.7b")
plt.savefig("figs/chain0.svg", format="svg", bbox_inches="tight")


model_name = 'gpt3-6.7b'
batch_size=16
num_heads=32
input_format='opt_schedules_mm'

# Get workload chain specs
chains, layers_dict, _ = utils.get_chain_config(workload_dir, output_dir, model_name=model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size)
# Calculate theoretical optimal bounds
optimal_accesses, optimal_accesses_fused = plots.get_optimal_performance(chains, layers_dict, num_heads)

# 6 GEMMs is the 0th chain in the gpt3-6.7b workload
chain_idx = 0

# Get path name to different bounds
csv_unfused = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size,
    constraint_config='', scheme='opt', enable_fusion=False,  matmul_only=False)
csv_tiled_fusion = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size
    )
csv_tiled_segmented_fusion = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size, 
    eval_slices=True)
csv_tiled_segmented_fusion_flash = utils.get_output_path(
    chain_idx, output_dir, model_name, input_format=input_format, num_heads=num_heads, batch_size=batch_size, 
    constraint_config='_relax_io_kn_flash', eval_slices=True)

df_unfused = pd.read_csv(csv_unfused)
df_tiled_fusion =  pd.read_csv(csv_tiled_fusion)
df_tiled_segmented_fusion = pd.read_csv(csv_tiled_segmented_fusion)
df_tiled_fusion_flash = pd.read_csv(csv_tiled_fusion_flash)
df_tiled_segmented_fusion_flash = pd.read_csv(csv_tiled_segmented_fusion_flash)

ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx],
                                    df_nochain=(df_unfused, 'No Fusion'), 
                                    df_slice=(df_tiled_segmented_fusion_flash, 'Segmented Tiled Fusion'), 
                                    df_relax_io=(df_tiled_fusion_flash, 'Tiled Fusion Flash'),
                                    df_relax_io_kn=(df_tiled_segmented_fusion_flash, 'Segmented Tiled Fusion Flash'),
                                    xbound=(10**4, None), ybound=(None, 10**12), logx=True, logy=True,
                                    figsize=(3, 2.5), max_effect_size=True, y_max_effectual=0.001,
                                    plot_cache=['100MB'])
legend = ax.legend(loc='upper center', bbox_to_anchor=(1.5, 1), fontsize=8)
ax.set_title("opt_sched_mm, 0th chain (6 GEMMS) in gpt3-6.7b")
plt.savefig("figs/optchain0.svg", format="svg", bbox_inches="tight")
