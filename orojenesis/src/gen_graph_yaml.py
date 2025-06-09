import networkx as nx
from collections import OrderedDict
import yaml
import pathlib
import argparse
from . import utils
from . import graph_utils
from . import gen_prob
import os

TIMELOOP_BASE_PATH = pathlib.Path(os.getenv("TIMELOOP_BASE_PATH"))
logger = utils.logger

def load_model_config(model_name='gpt3-6.7b'):
    """Load model configuration from YAML file."""
    config_path = TIMELOOP_BASE_PATH / 'orojenesis' / 'configs' / 'workloads' / f'{model_name}.yaml'
    
    if not config_path.exists():
        raise ValueError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Generate GPT computation graph')
    parser.add_argument('--model', type=str, default='gpt3-6.7b', 
                       help='Model name (default: gpt3-6.7b)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--matmul-only', action='store_true', default=False,
                       help='Generate matmul-only version')
    parser.add_argument('--nored', type=str, default='nored', choices=['', 'nored', 'noln'],
                       help='Reduction options (default: nored)')
    parser.add_argument('--output-dir', type=str, default= TIMELOOP_BASE_PATH/'orojenesis'/'workloads',
                       help='Output directory (default: orojenesis/workloads)')
    
    args = parser.parse_args()
    
    # Load model configuration
    config = load_model_config(args.model)
    
    config_dir = TIMELOOP_BASE_PATH / 'orojenesis' / 'configs'  
    # Extract configuration parameters
    model_name = config['model_name']
    seq_len = config['seq_len']
    input_seq_len = config['input_seq_len']
    num_heads = config['num_heads']
    d_proj = config['d_proj']
    d_fc = config['d_fc']
    d = config['d']

    # Set parameters from command line arguments
    matmul_only = args.matmul_only
    nored = args.nored
    batch = args.batch
    
    postfix = "_matmul" if matmul_only else ""
    postfix1 = f"_{nored}" if nored != "" else ""
    postfix2 = f"_b{batch}" # if batch > 1 else ""
    folder_name = f'{model_name}_graph{postfix2}{postfix}{postfix1}'
    output_dir = TIMELOOP_BASE_PATH / 'orojenesis' / 'workloads' / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'output_dir: {output_dir}')

     # Find parent and children nodes

    # Load the graph YAML file
    graph_yaml_path = config_dir / 'workloads' / f'{folder_name}.yaml'
    with open(graph_yaml_path, 'r') as f:
        graph_data = yaml.safe_load(f)

    # Create a new directed graph
    G = nx.DiGraph()

    new_nodes = OrderedDict()
    # Add nodes with their properties
    for node_info in graph_data:
        node_name = node_info['name']
        new_nodes[node_name] = node_info['properties']
        G.add_node(node_name, **node_info['properties'])

    # Add edges based on predecessors
    for node_info in graph_data:
        node_name = node_info['name']
        for pred in node_info['predecessors']:
            G.add_edge(pred, node_name)

    # Verify the graph structure
    logger.info(f'New graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    
    # Optional: Compare with original graph
    if G.number_of_nodes() == G.number_of_nodes() and G.number_of_edges() == G.number_of_edges():
        logger.info('Graph reconstruction successful - node and edge counts match')
    else:
        logger.warning('Graph reconstruction may have issues - node/edge counts differ from original')


    for node in G.nodes():
        logger.info(f'Node: {node}')
        logger.info(f'Properties: {G.nodes[node]}')
        logger.info(f'Predecessors: {G.predecessors(node)}')
        logger.info(f'Successors: {G.successors(node)}')


    layers_dict={}
    layers=[]
    sources = []
    targets = []
    # dump mm yaml to calculate oaves for each layer
    # logger.info(f'Nodes: {nodes.keys()}')
    for k,v in new_nodes.items():
        print(f'Node: {k}, Properties: {v}')

        if v['source']:
            sources.append(k)
        if v['target']:
            targets.append(k)

        if 'type' not in v:
            continue

        if v['type'] in ['mm', 'bmm']:
            prob_name = gen_prob.gen_yaml(config_dir / v['template'], str(output_dir), v['shape'])
            layers_dict[k] = prob_name
            layers.append(prob_name)
        elif v['type'] in ['ew', 'red']:
            v_type = v['type']
            prob_name = gen_prob.gen_yaml(config_dir / v['template'], str(output_dir), v['shape'], prefix=f'{v_type}_')
            layers_dict[k] = prob_name
            layers.append(prob_name)
        

    utils.store_yaml(output_dir / 'layers.yaml', layers)
    utils.store_yaml(output_dir / 'layers_dict.yaml', layers_dict)


    G_mm = G.copy()

    not_modeled_ops = [f'split_{i}' for i in range(3)] + ['transpose', 'concat', 'gelu', 'ewsingle_softmax_exp']

    for node in G.nodes():
        if node in not_modeled_ops or 'ew_add' in node or 'red_layernorm' in node or 'red_softmax' in node:
            graph_utils.remove_node(G_mm, node)

    p=nx.drawing.nx_pydot.to_pydot(G_mm)
    p.set_rankdir("LR")
    p.write_pdf(output_dir/ f'model.pdf')

    paths = []

    for source in sources:
        paths.extend(list(nx.all_simple_paths(G_mm, source=source, target=targets)))

    logger.info(f'{len(paths)} simple paths generated')
    utils.store_yaml(output_dir / 'paths.yaml', paths)
    logger.info(f'Paths: {paths}')

    # define the reduction op that will break mm chaining
    # reduction_ops = ['softmax', 'layernorm']
    if nored == "":
        reduction_ops = ['softmax', 'layernorm']
    elif nored == "nored":
        reduction_ops = []
    elif nored == "noln":
        reduction_ops = ['softmax']
    else:
        raise("Invalid spec for nored parameter")

    # generate op chains
    chains = []
    for path in paths:
        chain = []
        chainable_ops = []
        for op in path:
            if 'mm' in op or 'bmm' in op:
                chainable_ops.append(op)
            for reduction_op in reduction_ops:
                if reduction_op in op:
                    if len(chainable_ops) > 1:
                        chain.append(chainable_ops)
                    chainable_ops= []
        if len(chainable_ops) > 1:
            chain.append(chainable_ops)
        logger.info(f'Chain: {chain}')
        if chain != []:
            chains.append(chain)

    # chains = graph_utils.remove_duplicate_paths(chains)
    utils.store_yaml(output_dir / 'chains.yaml', chains)

    # generate op chains
    pair_chains = []
    for chain in chains:
        pair_chain = []
        for chainable_ops in chain:
            op_pair = [chainable_ops[0]]
            for op in chainable_ops[1:]:
                op_pair.append(op)
                pair_chain.append(op_pair)
                op_pair = [op]
        pair_chains.append(pair_chain)
    utils.store_yaml(output_dir / 'pair_chains.yaml', pair_chains)


    # generate different chain slices
    graph_utils.generate_slices(chains, output_dir, prefix='chain')

    not_modeled_ops = [f'split_{i}' for i in range(3)] + ['transpose', 'concat', 'gelu', 'ewsingle_softmax_exp']
    not_modeled_ops += ['input', 'output']

    for node in not_modeled_ops:
        if G.has_node(node):
            graph_utils.remove_node(G, node)

    all_schedules, optimal_chains = graph_utils.generate_schedules(G)
    utils.store_yaml(output_dir / 'schedules.yaml', all_schedules)
    logger.info(f'Schedules yaml: {output_dir / "schedules.yaml"}')
    utils.store_yaml(output_dir / 'opt_schedules.yaml', optimal_chains)
    graph_utils.generate_slices(optimal_chains, output_dir, prefix='opt_schedules')


    for node in list(G.nodes()):
        if 'bmm' not in node and 'mm' not in node:
            graph_utils.remove_node(G, node)

    p=nx.drawing.nx_pydot.to_pydot(G)
    p.set_rankdir("LR")
    p.write_pdf(output_dir/ 'model_pruned.pdf')
    all_schedules, optimal_chains = graph_utils.generate_schedules(G)
    utils.store_yaml(output_dir / 'schedules_mm.yaml', all_schedules)
    logger.info(f'Schedules mm yaml: {output_dir / "schedules_mm.yaml"}')
    utils.store_yaml(output_dir / 'opt_schedules_mm.yaml', optimal_chains)
    graph_utils.generate_slices(optimal_chains, output_dir, prefix='opt_schedules_mm')


if __name__ == "__main__":
    main() 