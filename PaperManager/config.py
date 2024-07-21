import os
from utils import *

def mkdaf(path):
    if not os.path.exists(path):
        name = os.path.basename(path)
        name_slist = name.split('.')
        assert len(name_slist) <= 2
        if len(name_slist) == 2:
            assert name_slist[-1] == 'json'
            save_json({}, path)
        else:
            os.makedirs(path)



'''
path(root_path):
    list.json
    Papers/
'''
db_name = 'db_papers'
ROOT_PATH = './Database'
PAPER_ROOT_PATH = f'{ROOT_PATH}/Papers'


[mkdaf(e) for e in [ROOT_PATH, f'{ROOT_PATH}/{db_name}.json', PAPER_ROOT_PATH]]