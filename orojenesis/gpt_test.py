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
import src.gen_mappings as gen_mappings
import src.process_untiled_fusion as process_untiled_fusion
import src.process_tiled_fusion as process_tiled_fusion
import time

workload_dir = pathlib.Path('./workloads')
config_dir = pathlib.Path('./configs/multi-einsum')
output_dir = pathlib.Path('./outputs/multi-einsum')

# Multi-Einsum Mapping Search for GPT-6.7b in Fig.18, 21-23. 
# It will take roughly 2 hours to finish all experiments on a 4-core Intel® Core™i7-1185G7 processor @ 3.00 GHz.
model_name = 'gpt3-6.7b'
batch_size = 16
num_heads = 32
arch_prefix = ''
force_rerun = False



# Generate baseline optimal unconstrained unfused mappings.
start_time = time.time()
gen_mappings.gen_mappings(workload_dir, config_dir, output_dir,
                          model_name=model_name, batch=batch_size, num_heads=num_heads,
                          spatial_factor=None, arch_prefix=arch_prefix, ffmt=False,
                          force_rerun=force_rerun)
print(f"Time to generate unconstrained unfused mappings: {(time.time() - start_time):.2f}")

# Generate mappings following fusion friendly mapping templates (FFMT)
start_time = time.time()
gen_mappings.gen_mappings(workload_dir, config_dir, output_dir, \
                          model_name=model_name, batch=batch_size, num_heads=num_heads,
                          spatial_factor=None, arch_prefix=arch_prefix, ffmt=True,
                          force_rerun=force_rerun)
print(f"Time to generate fusion friendly mappings (FFMT): {(time.time() - start_time):.2f}")

###

# For six-layer chain
# Generate the unfused, and untiled fusion bounds
start_time = time.time()
process_untiled_fusion.process_untiled_fusion(workload_dir, output_dir, model_name=model_name,
                                              input_format='chain', num_heads=num_heads,
                                              batch=batch_size, arch_prefix=arch_prefix,
                                              matmul_only=False)
print(f"Time to generate untiled-unfused, flashattn-off, segm-chain-off, 6-layer-chain fusion bounds: {(time.time() - start_time):.2f}")

# w/o flashattn mapspace
# Generate the tiled fusion bounds
start_time = time.time()
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='chain', num_heads=num_heads,
                                          batch=batch_size, arch_prefix=arch_prefix)
print(f"Time to generate tiled, flashattn-off, segm-chain-off, 6-layer-chain fusion bounds: {(time.time() - start_time):.2f}")

# Generate the tiled fusion bounds for segmented chains
start_time = time.time()
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='chain', num_heads=num_heads,
                                          batch=batch_size, arch_prefix=arch_prefix,
                                          eval_slices=True)
print(f"Time to generate tiled, flashattn-off, segm-chain-on, 6-layer-chain fusion bounds: {(time.time() - start_time):.2f}")

# w/ flashattn mapspace
# Generate the tiled fusion bounds
start_time = time.time()
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='chain', num_heads=num_heads,
                                          batch=batch_size, arch_prefix=arch_prefix,
                                          constraint_config='_relax_io_kn_flash')
print(f"Time to generate tiled, flashattn-on, segm-chain-off, 6-layer-chain fusion bounds: {(time.time() - start_time):.2f}")

# Generate the tiled fusion bounds for segmented chains
start_time = time.time()
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='chain', num_heads=num_heads,
                                          batch=batch_size, arch_prefix=arch_prefix,
                                          eval_slices=True, constraint_config='_relax_io_kn_flash')
print(f"Time to generate tiled, flashattn-on, segm-chain-on, 6-layer-chain fusion bounds: {(time.time() - start_time):.2f}")


# For full LLM block schedule
# Generate the unfused, and untiled fusion bounds
start_time = time.time()
process_untiled_fusion.process_untiled_fusion(workload_dir, output_dir, model_name=model_name,
                                              input_format='opt_schedules_mm', num_heads=num_heads,
                                              batch=batch_size, arch_prefix=arch_prefix,
                                              matmul_only=False)
print(f"Time to generate untiled-unfused, flashattn-off, segm-chain-off, full-LLM-block fusion bounds: {(time.time() - start_time):.2f}")

# Generate the tiled fusion bounds
start_time = time.time()
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='opt_schedules_mm', num_heads=num_heads,
                                          batch=batch_size, arch_prefix=arch_prefix)
print(f"Time to generate tiled, flashattn-off, segm-chain-off, full-LLM-block fusion bounds: {(time.time() - start_time):.2f}")

# Generate the tiled fusion bounds for segmented chains
start_time = time.time()
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='opt_schedules_mm', num_heads=num_heads,
                                          batch=batch_size, arch_prefix=arch_prefix,
                                          eval_slices=True)
print(f"Time to generate tiled, flashattn-off, segm-chain-on, full-LLM-block fusion bounds: {(time.time() - start_time):.2f}")

# Generate the tiled fusion bounds for segmented chains w/ flashattn
start_time = time.time()
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='opt_schedules_mm', num_heads=num_heads,
                                          batch=batch_size, arch_prefix=arch_prefix,
                                          eval_slices=True, constraint_config='_relax_io_kn_flash')
print(f"Time to generate tiled, flashattn-on, segm-chain-on, full-LLM-block fusion bounds: {(time.time() - start_time):.2f}")



########


# Multi-Einsum Mapping Search for MHA in Fig.20. 
# It will take roughly 5 minutes to finish all experiments on a 4-core Intel® Core™i7-1185G7 processor.
model_name = "attn-block"
batch_size = 16
num_heads = 32
force_rerun = False

start_time = time.time()
gen_mappings.gen_mappings(workload_dir, config_dir, output_dir, model_name=model_name, batch=batch_size, num_heads=num_heads,
                          spatial_factor=None, arch_prefix='', ffmt=False, force_rerun=force_rerun)
gen_mappings.gen_mappings(workload_dir, config_dir, output_dir, model_name=model_name, batch=batch_size, num_heads=num_heads,
                          spatial_factor=None, arch_prefix='', ffmt=True, force_rerun=force_rerun)

process_untiled_fusion.process_untiled_fusion(workload_dir, output_dir, model_name=model_name,
                                              input_format='chain', num_heads=num_heads, batch=batch_size,
                                              matmul_only=False)
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='chain', num_heads=num_heads, batch=batch_size)
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='chain', num_heads=num_heads, batch=batch_size,
                                          constraint_config='_relax_io')
process_tiled_fusion.process_tiled_fusion(workload_dir, output_dir, model_name=model_name,
                                          input_format='chain', num_heads=num_heads, batch=batch_size,
                                          constraint_config='_flash')
print(f"Time to generate all attn block fusion bounds: {(time.time() - start_time):.2f}")

########


fig_dir = pathlib.Path('./figs')
fig_dir.mkdir(parents=True, exist_ok=True)


########

start_time = time.time()

model_name = 'gpt3-6.7b'
batch_size=16
num_heads=32
input_format='chain'

# Get workload chain specs
chains, layers_dict, _ = utils.get_chain_config(workload_dir, output_dir, model_name=model_name, 
                                                    input_format=input_format, num_heads=num_heads, batch_size=batch_size)
for chain_idx, chain in enumerate(chains):
    print(f'chain {chain_idx}: {chain}')
    
# Calculate theoretical optimal bounds 
optimal_accesses, optimal_accesses_fused = plots.get_optimal_performance(chains, layers_dict, num_heads)

# FFN is the 4th chain in the gpt3-6.7b workload 
chain_idx = 4 

# Get path name to different bounds 
csv_tiled_fusion = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         num_heads=num_heads, batch_size=batch_size)
csv_unfused = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         constraint_config='', scheme='opt', num_heads=num_heads, batch_size=batch_size, enable_fusion=False, matmul_only=False)
csv_untiled_fusion = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         constraint_config='', scheme='opt', num_heads=num_heads, batch_size=batch_size)
csv_tiled_fusion_flash = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         num_heads=num_heads, batch_size=batch_size, constraint_config='_relax_io_kn_flash')

# Load bounds to pandas dataframe 
df_tiled_fusion =  pd.read_csv(csv_tiled_fusion)      
df_unfused = pd.read_csv(csv_unfused)
df_untiled_fusion = pd.read_csv(csv_untiled_fusion)
df_tiled_fusion_flash = pd.read_csv(csv_tiled_fusion_flash)

ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], \
                                    optimal_accesses_fused[chain_idx],  figsize=(2.5,2.5), \
                                    df_nochain=(df_unfused, 'No Fusion'), \
                                    df_nochain_fused=(df_untiled_fusion, 'Untiled Fusion'), \
#                                     df_relax_io=(df_tiled_fusion_flash, 'Tiled Fusion Flash'), \
                                    xbound=(10**3, 2*10**9), ybound=(None, 8*10**11), max_effect_size=None)
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.6, 1), fontsize=8)
plt.savefig(f"{fig_dir}/fig18a.pdf", format="pdf", bbox_inches="tight") 

# calculate the reduction stats of the tiled fusion and unfused bounds
df_red = utils.compute_reduction(df_unfused, df_tiled_fusion)
fig, ax = plt.subplots(dpi=300, figsize=(2.5, 2.5))
plt.ylabel('Reduction Factor \n (No fusion / Tiled fusion)')        
line = ax.axhline(y=1, color='black', linestyle='-', lw=1, label='SOL w/o fusion')
df_red.set_index('max_buf_size').sort_index()['accesses_ratio'].plot(ax=ax, logx=True, xlim=(10**3, 10**9))
plt.xlabel('Buffer Size (B)')
plt.savefig(f"{fig_dir}/fig18b.pdf", format="pdf", bbox_inches="tight")

model_name = 'attn-block'
batch_size=16
num_heads=32
input_format='chain'

# Get workload chain specs
chains, layers_dict, _ = utils.get_chain_config(workload_dir, output_dir, model_name=model_name, 
                                                    input_format=input_format, num_heads=num_heads, batch_size=batch_size)

# Calculate theoretical optimal bounds 
optimal_accesses, optimal_accesses_fused = plots.get_optimal_performance(chains, layers_dict, num_heads, batch_size)
print(f'Data reduction: {optimal_accesses[0]/optimal_accesses_fused[0]}')

# MHA is the 0th chain
chain_idx = 0

# Get path name to different bounds 
csv_tiled_fusion_kn = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         num_heads=num_heads, batch_size=batch_size)
csv_tiled_fusion_n = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         constraint_config='_relax_io', num_heads=num_heads, batch_size=batch_size)
csv_tiled_fusion_flash = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         constraint_config='_flash', num_heads=num_heads, batch_size=batch_size)
csv_unfused = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         constraint_config='', scheme='opt', num_heads=num_heads, batch_size=batch_size, enable_fusion=False, matmul_only=False)

df_tiled_fusion_kn = pd.read_csv(csv_tiled_fusion_kn)
df_tiled_fusion_n = pd.read_csv(csv_tiled_fusion_n)
df_tiled_fusion_flash = pd.read_csv(csv_tiled_fusion_flash)
df_unfused = pd.read_csv(csv_unfused)
ax = plots.plot_accesses_comparison((df_tiled_fusion_flash, 'FlashAttention'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx], 
                                    df_nochain=(df_unfused, 'No Fusion'), df_relax_io=(df_tiled_fusion_n, 'FLAT'), 
                                    figsize=(5*2.5/3,2.5), xbound=(10**2, None), ybound=(None, None)) # (df_tiled_fusion_kn, 'tileKN')
legend = ax.legend(loc='upper left', ncol=3, fontsize=8)
plt.savefig(f"{fig_dir}/fig20.pdf", format="pdf", bbox_inches="tight")

model_name = 'gpt3-6.7b'
batch_size=16
num_heads=32
input_format='chain'
arch_prefix=''

# Get workload chain specs
chains, layers_dict, _ = utils.get_chain_config(workload_dir, output_dir, model_name=model_name, 
                                                    input_format=input_format, num_heads=num_heads, batch_size=batch_size)
# Calculate theoretical optimal bounds 
optimal_accesses, optimal_accesses_fused = plots.get_optimal_performance(chains, layers_dict, num_heads, batch_size)

# 6 GEMMs is the 0th chain in the gpt3-6.7b workload 
chain_idx = 0

# Get path name to different bounds 
csv_tiled_fusion = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                   num_heads=num_heads, batch_size=batch_size, arch_prefix=arch_prefix)
csv_unfused = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         constraint_config='', scheme='opt', num_heads=num_heads, batch_size=batch_size,
                                         arch_prefix=arch_prefix, enable_fusion=False, matmul_only=False)
csv_tiled_segmented_fusion = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         num_heads=num_heads, batch_size=batch_size, arch_prefix=arch_prefix, eval_slices=True)
csv_tiled_segmented_fusion_flash = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         num_heads=num_heads, batch_size=batch_size, arch_prefix=arch_prefix, eval_slices=True,
                                                   constraint_config='_relax_io_kn_flash')

df_tiled_fusion =  pd.read_csv(csv_tiled_fusion)      
df_unfused = pd.read_csv(csv_unfused)
df_tiled_segmented_fusion = pd.read_csv(csv_tiled_segmented_fusion)
df_tiled_segmented_fusion_flash = pd.read_csv(csv_tiled_segmented_fusion_flash)

df_csv = df_tiled_segmented_fusion_flash[["max_buf_size", "fused_accesses", "slice", "mapping"]]
df_csv.to_csv(f'{model_name}_b{batch_size}_fused.csv')

# log-scale
ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx], 
                                    figsize=(5*2.5/3,3*2.5/3), logx=True, logy=True, df_nochain=(df_tiled_segmented_fusion, 'No Fusion'), 
                                    xbound=(10**3, None), ybound=(None, None), 
                                    df_slice=(df_tiled_segmented_fusion_flash, 'Segmented Tiled Fusion'), max_effect_size=True) # (df_tiled_segmented_fusion, 'Segmented Tiled Fusion')
legend = ax.legend(loc='upper center', ncol=2, columnspacing=1, fontsize=8)

# log-scale
ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx], 
                                    figsize=(6*2.5/3,3*2.5/3), logx=True, logy=True, df_nochain=(df_tiled_segmented_fusion, 'No Fusion'), 
                                    xbound=(10**3, None), ybound=(None, None), max_effect_size=True) # (df_tiled_segmented_fusion, 'Segmented Tiled Fusion')
legend = ax.legend(loc='upper center', ncol=2, columnspacing=1, fontsize=8)

# linear-scale 
ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx], 
                                    figsize=(5*2.5/3,3*2.5/3), logx=False, logy=False, df_nochain=(df_unfused, 'No Fusion'), xbound=(10**6, None), ybound=(0, 10*10**10), 
                                    df_slice=(df_tiled_segmented_fusion_flash, 'Segmented Tiled Fusion'), max_effect_size=False) # (df_tiled_segmented_fusion, 'Segmented Tiled Fusion')
legend = ax.legend(loc='upper center', ncol=2, columnspacing=1, fontsize=8)
 
ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx], 
                                    figsize=(8*2.5/3,2*2.5/3), logx=False, logy=False, df_nochain=None, xbound=(1**6, None), ybound=(0, 1*10**10), 
                                    df_slice=(df_tiled_segmented_fusion_flash, 'Segmented Tiled Fusion'), max_effect_size=False) # (df_tiled_segmented_fusion, 'Segmented Tiled Fusion')

legend = ax.legend(loc='upper center', ncol=2, columnspacing=1, fontsize=8)
plt.savefig("figs/fig21.pdf", format="pdf", bbox_inches="tight")

model_name = 'gpt3-6.7b'
batch_size=16
num_heads=32
input_format='opt_schedules_mm'

# Get workload chain specs
chains, layers_dict, _ = utils.get_chain_config(workload_dir, output_dir, model_name=model_name, 
                                                    input_format=input_format, num_heads=num_heads, batch_size=batch_size)
# Calculate theoretical optimal bounds 
optimal_accesses, optimal_accesses_fused = plots.get_optimal_performance(chains, layers_dict, num_heads)

# 6 GEMMs is the 0th chain in the gpt3-6.7b workload 
chain_idx = 0

# Get path name to different bounds 
csv_tiled_fusion = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                   num_heads=num_heads, batch_size=batch_size)
csv_unfused = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         constraint_config='', scheme='opt', num_heads=num_heads, batch_size=batch_size, enable_fusion=False,  matmul_only=False)
csv_tiled_segmented_fusion = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         num_heads=num_heads, batch_size=batch_size, eval_slices=True)
csv_tiled_segmented_fusion_flash = utils.get_output_path(chain_idx, output_dir, model_name, input_format=input_format, 
                                         num_heads=num_heads, batch_size=batch_size, eval_slices=True, constraint_config='_relax_io_kn_flash')

df_tiled_fusion =  pd.read_csv(csv_tiled_fusion)      
df_unfused = pd.read_csv(csv_unfused)
df_tiled_segmented_fusion = pd.read_csv(csv_tiled_segmented_fusion)
df_tiled_segmented_fusion_flash = pd.read_csv(csv_tiled_segmented_fusion_flash)

ax = plots.plot_accesses_comparison((df_tiled_fusion, 'Tiled Fusion'), optimal_accesses[chain_idx], optimal_accesses_fused[chain_idx], 
                                    figsize=(5*2.5/3,2.5), df_nochain=(df_unfused, 'No Fusion'), xbound=(10**6, 5*10**8), ybound=(None, 10**11), 
                                    df_slice=(df_tiled_segmented_fusion_flash, 'Segmented Tiled Fusion'), max_effect_size=True, y_max_effectual=0.001,
                                    x_algo_min_unfused=3, x_algo_min_fused=3, plot_cache=['L2'])
legend = ax.legend(loc='upper center', ncol=2, columnspacing=1, fontsize=8)
plt.savefig("figs/fig22.pdf", format="pdf", bbox_inches="tight")
min_accesses = df_tiled_fusion['fused_accesses'].min()
print(f'Optimal data movement reduction: {optimal_accesses[chain_idx]  / min_accesses}')

#  Arch specs 
area_per_mac = 332.25  # um^2
area_per_B = 2.59122   # um^2
chip_area = 529*10**6 # um^2
freq = 0.7 * 10**9     # GHz
bw_Bps=147.8*10**9     # B/s

total_area = chip_area * 0.8 # um^2, we assume 80% of chip area is dedicated to the MAC and buffer
total_compute=14399012077568.0 # MACs

# Workload specs 
model_name = 'gpt3-6.7b'
batch_size=16
num_heads=32
input_format='opt_schedules_mm'

# Get path name to different bounds 
csv_untiled = utils.get_output_path(0, output_dir, model_name, input_format=input_format, 
                                         constraint_config='', scheme='opt', num_heads=num_heads, batch_size=batch_size, enable_fusion=False)
csv_tiled_segmented_fusion = utils.get_output_path(0, output_dir, model_name, input_format=input_format, 
                                         num_heads=num_heads, batch_size=batch_size, eval_slices=True, constraint_config='_relax_io_kn_flash')
df_untiled = pd.read_csv(csv_untiled)
df_tiled_segmented_fusion = pd.read_csv(csv_tiled_segmented_fusion)

# Note that please the previous cell before running this one 
mem_bound_perf, compute_bound_perf, perf, buf_ratio, intersection = utils.derive_performance_bounds(df_untiled, area_per_mac, area_per_B, freq, bw_Bps, total_area, total_compute)
print(max(perf))
plots.plot_buf_area_tradeoff(mem_bound_perf, compute_bound_perf, perf, buf_ratio, intersection, area_per_B, total_area, plot_offset=-0.36, figname='design_llm_unfused')

mem_bound_perf, compute_bound_perf, perf, buf_ratio, intersection = utils.derive_performance_bounds(df_tiled_segmented_fusion, area_per_mac, area_per_B, freq,  bw_Bps, total_area, total_compute)
print(max(perf))
plots.plot_buf_area_tradeoff(mem_bound_perf, compute_bound_perf, perf, buf_ratio, intersection, area_per_B, total_area, plot_offset=0.02, figname='design_llm_fused')

print(f"Time to generate plots: {(time.time() - start_time):.2f}")

