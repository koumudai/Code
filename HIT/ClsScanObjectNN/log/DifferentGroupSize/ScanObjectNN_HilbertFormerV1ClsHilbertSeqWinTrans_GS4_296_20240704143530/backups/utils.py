import os
import yaml
import random
import numpy as np
import torch
import datetime


# set random seed 
def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(10)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True 


# io
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def print(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def generate_exp_dir(cfg, seed):
    root_path, exp_name, save_file_list = cfg.exp_root_path, f'{cfg.exp_name}_{seed}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}', cfg.save_file_list
    if not os.path.exists(f'{root_path}/{exp_name}/backups'):
        os.makedirs(f'{root_path}/{exp_name}/backups')

    backups_path = f'{root_path}/{exp_name}/backups'
    for base_path in ['/'.join(e.split('/')[:-1]) for e in save_file_list if '/' in e]:
        if not os.path.exists(f'{backups_path}/{base_path}'):
            os.makedirs(f'{backups_path}/{base_path}')
    for save_path in save_file_list:
        os.system(f'cp {save_path} {backups_path}/{save_path}')
    return f'{root_path}/{exp_name}'


def load_config(cfg_path):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    with open(cfg_path) as f:
       cfg = yaml.safe_load(f)
    return cfg