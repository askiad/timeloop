import argparse
import yaml
import pathlib
from . import utils

logger = utils.logger

def construct_argparser():
    """ Returns argument parser """
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-c',
                        '--configs',
                        nargs='+',
                        help='Problem Dims M K N',
                        )
    parser.add_argument('-i',
                        '--template_yaml',
                        type=str,
                        default='prob_template.yaml',
                        help='Problem Template YAML',
                        )
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='./',
                        help='Output Dir',
                        )

    return parser


def gen_yaml(template_yaml: str, output_dir: str, configs: list, prefix: str="prob_"):
    """ Generate problem config.

    Args:
	template_yaml: yaml template
	output_dir: output dir path to store output yaml
        configs: list of problem dimensions
    """
    with open(template_yaml, 'r') as f:
        template = yaml.safe_load(f)

    instance = template['problem']
    if 'instance' in instance.keys():
        instance = instance['instance']

    sizes = []
    for config in configs:
        items = config.split('=')
        prob_dim = items[0]
        size = int(items[1])
        sizes.append(items[1])
        instance[prob_dim] = size

    output_dir = pathlib.Path(output_dir)
    name = prefix + ('_').join(sizes)
    filename = name + '.yaml'
    output_file = output_dir / filename
    logger.info(f"Generated prob yaml: {output_file}")
    with open(output_file, 'w') as f:
        yaml.dump(template, f, default_flow_style=False)

    return name

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    gen_yaml(args.template_yaml, args.output_dir, args.configs)
