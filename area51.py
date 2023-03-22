# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.utils import instantiate

class Test:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def printself(self):
        print(self.x, self.y)

@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    for key, obj in cfg.list.items():
        print(key)
        instantiate(obj).printself()
    instantiate(cfg.obj).printself()


if __name__ == "__main__":
    my_app()