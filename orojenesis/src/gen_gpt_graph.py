import networkx as nx
from collections import OrderedDict
import yaml
import pathlib
import argparse
from . import utils
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


def parse_arguments():
    """Parse command line arguments."""
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
    
    return parser.parse_args()


def create_nodes(config, batch, matmul_only):
    """Create nodes dictionary based on configuration and parameters."""
    seq_len = config['seq_len']
    input_seq_len = config['input_seq_len']
    num_heads = config['num_heads']
    d_proj = config['d_proj']
    d_fc = config['d_fc']
    d = config['d']
    
    nodes = OrderedDict()
    chaining_helper_ops = OrderedDict()

    # Addition op type for chaining
    for i in range(num_heads):
        chaining_helper_ops[f'mm_proj_split_{i}'] = [seq_len * batch, d_proj, d//num_heads]
        chaining_helper_ops[f'transpose_{i}'] = [d//num_heads, seq_len * batch] # transpose k
        chaining_helper_ops[f'mm_qk_{i}'] = [seq_len * batch, d//num_heads, seq_len] # QK act-to-act
        chaining_helper_ops[f'softmax_{i}'] = []
        chaining_helper_ops[f'mm_qkv_{i}'] = [seq_len * batch, seq_len, d//num_heads] # QKV act-to-act

    nodes['input'] = []
    for i in range(3):
        if matmul_only:
            nodes[f'mm_proj_{i}'] = [seq_len * batch, d_proj, d//num_heads] # QKV proj
        else:
            nodes[f'mm_proj_{i}'] = [seq_len * batch, d_proj, d] # QKV proj
        nodes[f'split_{i}'] = [seq_len * batch, d//num_heads]

    nodes[f'transpose'] = [] # transpose k
    if matmul_only:
        nodes[f'mm_qk'] = [seq_len * batch, d//num_heads, input_seq_len] # QK act-to-act
        nodes[f'ewsingle_softmax_exp'] = [seq_len * batch, input_seq_len]
        nodes[f'red_softmax'] = [seq_len * batch, input_seq_len, 1]
        nodes[f'ew_softmax_div'] = [seq_len * batch, input_seq_len]
        nodes[f'mm_qkv'] = [seq_len * batch, input_seq_len, d//num_heads] # QKV act-to-act
    else:
        nodes[f'bmm_qk'] = [num_heads * batch, seq_len, d//num_heads, seq_len] # QK act-to-act
        nodes[f'ewsingle_softmax_exp'] = [seq_len * batch, input_seq_len]
        nodes[f'red_softmax'] = [seq_len * batch, input_seq_len, 1]
        nodes[f'ew_softmax_div'] = [seq_len * batch, input_seq_len]
        nodes[f'bmm_qkv'] = [num_heads * batch, seq_len, seq_len, d//num_heads] # QKV act-to-act

    nodes['concat'] = []
    if matmul_only:
        nodes['mm_proj_final'] = [seq_len * batch, d//num_heads, d_proj]
    else:
        nodes['mm_proj_final'] = [seq_len * batch, d, d_proj]
    
    nodes['red_layernorm1_mean'] = [seq_len * batch, d_proj, 1]
    nodes['red_layernorm1_var'] = [seq_len * batch, d_proj, 1]
    nodes['ew_layernorm1'] = [seq_len * batch, d_proj]
    nodes['ew_add1'] = [seq_len * batch, d_proj]
    nodes['mm_fc1'] = [seq_len * batch, d_proj, d_fc] # FC1
    nodes['gelu'] = []
    nodes['mm_fc2'] = [seq_len * batch, d_fc, d_proj] # FC2
    nodes['red_layernorm2_mean'] = [seq_len * batch, d_proj, 1]
    nodes['red_layernorm2_var'] = [seq_len * batch, d_proj, 1]
    nodes['ew_layernorm2'] = [seq_len * batch, d_proj]
    nodes['ew_add2'] = [seq_len * batch, d_proj]
    nodes['output'] = []
    nodes['attn_output'] = []

    return nodes, chaining_helper_ops


def build_adjacency_graph(nodes, model_name, matmul_only):
    """Build adjacency list representation of the graph."""
    adj = {}
    ki = {}
    for i, k in enumerate(nodes.keys()):
        ki[k] = k
        adj[k] = []

    # Determine sources and targets
    if 'attn-block' not in model_name:
        sources = ['input']
        targets = ['output']
    else:
        sources = [f'mm_qk']
        targets = ['concat']

    # Construct adjacency graph
    if 'attn-block' not in model_name:
        for i in range(3):
            adj[ki[f'input']].append(ki[f'mm_proj_{i}'])
            adj[ki[f'mm_proj_{i}']].append(ki[f'split_{i}'])

    if matmul_only:
        adj[ki[f'split_0']].append(ki[f'mm_qk'])
        adj[ki[f'split_1']].append(ki[f'transpose'])
        adj[ki[f'transpose']].append(ki[f'mm_qk'])
        adj[ki[f'mm_qk']].append(ki[f'ewsingle_softmax_exp'])
        adj[ki[f'ewsingle_softmax_exp']].append(ki[f'red_softmax'])
        adj[ki[f'red_softmax']].append(ki[f'ew_softmax_div'])
        adj[ki[f'ewsingle_softmax_exp']].append(ki[f'ew_softmax_div'])
        adj[ki[f'ew_softmax_div']].append(ki[f'mm_qkv'])
        adj[ki[f'split_2']].append(ki[f'mm_qkv'])
        adj[ki[f'mm_qkv']].append(ki['concat'])
    else:
        adj[ki[f'split_0']].append(ki[f'bmm_qk'])
        adj[ki[f'split_1']].append(ki[f'transpose'])
        adj[ki[f'transpose']].append(ki[f'bmm_qk'])
        adj[ki[f'bmm_qk']].append(ki[f'ewsingle_softmax_exp'])
        adj[ki[f'ewsingle_softmax_exp']].append(ki[f'red_softmax'])
        adj[ki[f'red_softmax']].append(ki[f'ew_softmax_div'])
        adj[ki[f'ewsingle_softmax_exp']].append(ki[f'ew_softmax_div'])
        adj[ki[f'ew_softmax_div']].append(ki[f'bmm_qkv'])
        adj[ki[f'split_2']].append(ki[f'bmm_qkv'])
        adj[ki[f'bmm_qkv']].append(ki['concat'])

    if 'attn-block' not in model_name:
        adj[ki[f'concat']].append(ki['mm_proj_final'])

        adj[ki[f'mm_proj_final']].append(ki['red_layernorm1_mean'])
        adj[ki[f'mm_proj_final']].append(ki['red_layernorm1_var'])
        adj[ki[f'red_layernorm1_mean']].append(ki['red_layernorm1_var'])
        adj[ki[f'mm_proj_final']].append(ki['ew_layernorm1'])
        adj[ki[f'red_layernorm1_var']].append(ki['ew_layernorm1'])

        adj[ki[f'ew_layernorm1']].append(ki['ew_add1'])
        adj[ki[f'input']].append(ki['ew_add1'])
        adj[ki[f'ew_add1']].append(ki['mm_fc1'])
        adj[ki[f'mm_fc1']].append(ki['gelu'])
        adj[ki[f'gelu']].append(ki['mm_fc2'])

        adj[ki[f'mm_fc2']].append(ki['red_layernorm2_mean'])
        adj[ki[f'mm_fc2']].append(ki['red_layernorm2_var'])
        adj[ki[f'red_layernorm2_mean']].append(ki['red_layernorm2_var'])
        adj[ki[f'mm_fc2']].append(ki['ew_layernorm2'])
        adj[ki[f'red_layernorm2_var']].append(ki['ew_layernorm2'])

        adj[ki[f'ew_layernorm2']].append(ki['ew_add2'])
        adj[ki[f'ew_add1']].append(ki['ew_add2'])
        adj[ki[f'ew_add2']].append(ki['output'])

    return adj, sources, targets


def create_networkx_graph(adj, nodes):
    """Create and process NetworkX graph."""
    G = nx.DiGraph()
    for k, v in adj.items():
        G.add_node(k)
        for node in v:
            G.add_edge(k, node)

    isolates = list(nx.isolates(G))  # Get a list of isolated nodes
    G.remove_nodes_from(isolates)

    for node in G.nodes:
        logger.info(f'Node: {node}')
        if 'input' not in node and 'output' not in node:
            G.nodes[node]['shape'] = 'box'
        if 'mm' in node:
            G.nodes[node]['label'] = f'{node}\n{nodes[node]}'
            G.nodes[node]['height'] = '1'

    return G


def create_nodes_dict(nodes, sources, targets):
    """Create nodes dictionary for YAML output."""
    nodes_dict = {}
    for k, v in nodes.items():
        nodes_dict[k] = {}
        nodes_dict[k]['source'] = k in sources
        nodes_dict[k]['target'] = k in targets
        
        if k.startswith('mm_'):
            nodes_dict[k]['type'] = 'mm'
            nodes_dict[k]['shape'] = [f'M={v[0]}', f'K={v[1]}', f'N={v[2]}']
            nodes_dict[k]['template'] = 'mm_template.yaml'
        elif k.startswith('bmm_'):
            nodes_dict[k]['type'] = 'bmm'
            nodes_dict[k]['shape'] = [f'H={v[0]}', f'M={v[1]}', f'K={v[2]}', f'N={v[3]}']
            nodes_dict[k]['template'] = 'bmm_template.yaml'
        elif k.startswith('ew_'):
            nodes_dict[k]['type'] = 'ew'
            nodes_dict[k]['shape'] = [f'M={v[0]}', f'K={v[1]}']
            nodes_dict[k]['template'] = 'ew_template.yaml'
        elif k.startswith('red_'):
            nodes_dict[k]['type'] = 'red'
            nodes_dict[k]['shape'] = [f'M={v[0]}', f'K={v[1]}']
            nodes_dict[k]['template'] = 'red_template.yaml'

    return nodes_dict


def generate_output(G, nodes, nodes_dict, config_dir, folder_name):
    """Generate final YAML output."""
    logger.info(f'Nodes: {nodes.keys()}')
    
    # Create a list to store node information
    node_list = []
    
    # Iterate through each node in the graph
    for node in nodes.keys():
        if node not in G.nodes():
            continue
        node_info = {
            'name': node,
            'properties': nodes_dict[node],
            'predecessors': list(G.predecessors(node)),
            'successors': list(G.successors(node))
        }
        node_list.append(node_info)
    
    # Store the node information in a YAML file
    utils.store_yaml(config_dir / 'workloads' / f'{folder_name}.yaml', node_list)
    logger.info(f'Node information stored in {config_dir}/workloads/{folder_name}.yaml')


def generate_gpt_graph(model='gpt3-6.7b', batch=16, matmul_only=True, nored='nored', output_dir=None):
    """
    Generate GPT computation graph.
    
    Args:
        model (str): Model name
        batch (int): Batch size
        matmul_only (bool): Generate matmul-only version
        nored (str): Reduction options
        output_dir (str): Output directory
    """
    # Load model configuration
    config = load_model_config(model)
    
    config_dir = TIMELOOP_BASE_PATH / 'orojenesis' / 'configs'  
    model_name = config['model_name']
    
    # Create folder name
    postfix = "_matmul" if matmul_only else ""
    postfix1 = f"_{nored}" if nored != "" else ""
    postfix2 = f"_b{batch}"
    folder_name = f'{model_name}_graph{postfix2}{postfix}{postfix1}'
    
    if output_dir is None:
        output_dir = TIMELOOP_BASE_PATH / 'orojenesis' / 'workloads' / folder_name
    else:
        output_dir = pathlib.Path(output_dir) / folder_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'output_dir: {output_dir}')

    # Create nodes
    nodes, chaining_helper_ops = create_nodes(config, batch, matmul_only)
    
    # Build adjacency graph
    adj, sources, targets = build_adjacency_graph(nodes, model_name, matmul_only)
    
    # Create NetworkX graph
    G = create_networkx_graph(adj, nodes)
    
    # Create nodes dictionary for YAML output
    nodes_dict = create_nodes_dict(nodes, sources, targets)
    
    # Generate output
    generate_output(G, nodes, nodes_dict, config_dir, folder_name)


def main():
    """Main function to handle command line interface."""
    args = parse_arguments()
    generate_gpt_graph(
        model=args.model,
        batch=args.batch,
        matmul_only=args.matmul_only,
        nored=args.nored,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main() 