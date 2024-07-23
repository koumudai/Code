import os
import shutil

if os.path.exists('log'):
    shutil.rmtree('log')
if os.path.exists('output'):
    shutil.rmtree('output')
os.makedirs('output/DifferentExpansion')