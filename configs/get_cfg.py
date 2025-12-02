import yaml

from configs.config import YAML_PATH

def get_cfg():
    with open(YAML_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
