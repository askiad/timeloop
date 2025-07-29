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
