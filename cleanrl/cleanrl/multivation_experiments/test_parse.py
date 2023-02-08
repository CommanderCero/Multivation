import yaml

config_path = "./extrinsic_only/four_heads_same_decay.yaml"
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)