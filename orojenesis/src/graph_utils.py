import networkx as nx
from itertools import product
from . import utils

logger = utils.logger

def remove_node(G, node_to_remove):
    children = list(G.successors(node_to_remove))
    parents = list(G.predecessors(node_to_remove))

    # Connect the parent node to each child node
    for parent in parents:
        for child in children:
            G.add_edge(parent, child)
            logger.info(f'ADD {parent} {child}')

    # Remove the node from the graph
    G.remove_node(node_to_remove)


def remove_duplicate_paths(list_of_list_of_list):
    unique_set = set(tuple(tuple(sublst) for sublst in lst_of_lst) for lst_of_lst in list_of_list_of_list)
    unique_list_of_list_of_list = [list(list(list(sublst) for sublst in tpl) for tpl in unique_set)]
    return unique_list_of_list_of_list


def partition_list(lst):
    def partition_helper(start, path):
        if start == len(lst):
            # Base case: All elements have been processed
            partitions.append(path[:])
            return

        for end in range(start + 1, len(lst) + 1): 
            # Generate all possible partitions from the current start index
            partition_helper(end, path + [lst[start:end]])

    partitions = []
    partition_helper(0, []) 
    return partitions


def prune_schedules(all_schedules, not_modeled_ops):
    pruned_all_schedules = []
    for schedule_idx, schedule in enumerate(all_schedules):
        pruned_schedule  = []
        for op in schedule:
            if op not in not_modeled_ops:
                pruned_schedule.append(op)
        pruned_all_schedules.append(pruned_schedule)

    unique_all_schedules = []
    seen_sublists = set()

    for sublist in pruned_all_schedules:
        sublist_tuple = tuple(sublist)
        if sublist_tuple not in seen_sublists:
            unique_all_schedules.append(sublist)
            seen_sublists.add(sublist_tuple)

    logger.info(f'Total {len(unique_all_schedules)} schedules.')
    return unique_all_schedules


# generate topological sorts
def generate_schedules(G):
    all_schedules = list(nx.all_topological_sorts(G))
    logger.info(f'Total {len(all_schedules)} schedules.')
    logger.info(all_schedules)

    schedule_idx_cc_tuple_list = []
    schedule_cc_list = []
    for schedule_idx, schedule in enumerate(all_schedules):
        # print(schedule)
        num_connected_components = 0
        for node_idx, node in enumerate(schedule):
            if node_idx == 0:
                prev_node = node
                cur_list = [[node]]
                continue
            if node in G.successors(prev_node):
                cur_list[-1].append(node)
                num_connected_components +=1
            else:
                cur_list.append([node])

            prev_node = node
        logger.info(f'Num connected components: {num_connected_components}')
        logger.info(f'Cur list: {cur_list}')
        schedule_cc_list.append(cur_list)
        schedule_idx_cc_tuple_list.append((schedule_idx, num_connected_components))

    # schedule_idx_cc_tuple_list = sorted(schedule_list, key=lambda x: x[1])
    max_value = max(schedule_idx_cc_tuple_list, key=lambda x: x[1])[1]
    max_list = [x for x in schedule_idx_cc_tuple_list if x[1] == max_value]

    optimal_chains = []
    for chain_idx, max_idx in enumerate(max_list):
        schedule_idx = max_idx[0]
        logger.info(f'Chain {chain_idx} {schedule_cc_list[schedule_idx]}')
        optimal_chains.append(schedule_cc_list[schedule_idx])
    return all_schedules, optimal_chains

def generate_slices(chains, output_dir, prefix='chain'):
    for chain_idx, chain in enumerate(chains):
        total_slices =  None
        for pair_idx, pair in enumerate(chain):
            slices=partition_list(pair)
            if total_slices is None:
                total_slices = slices
            else:
                total_slices = [a + b for a, b in product(total_slices, slices)]
        logger.info(f'Total number of slices = {len(total_slices)}')
        slice_path = output_dir / f'{prefix}{chain_idx}_slices.yaml'
        utils.store_yaml(slice_path, total_slices)
        logger.info(f'Path to slices files: {slice_path}')
        # logger.info(f'Slices: {utils.parse_yaml(slice_path)}')



