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

root_path = './DB'
backups_path = './backups'



'''
path(root_path):
    list.json
    papers/
    comments/
'''
path_list = ['Papers', 'Comments']
db_name = 'db_papers'
PAPER_ROOT_PATH = f'{root_path}/Papers'


[mkdaf(e) for e in [root_path, f'{root_path}/{db_name}.json', *[f'{root_path}/{e}' for e in path_list]]]