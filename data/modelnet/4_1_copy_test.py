"""
Split.
"""

import random
import shutil

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import shutil

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 4_split.py config_folder')
        exit(1)

    config_file = sys.argv[1] + '/test.json'
    assert os.path.exists(config_file)

    config = utils.read_json(config_file)

    simplified_directory = config['simplified_directory'] + '/'
    assert os.path.exists(simplified_directory), 'directory %s does not exist' % simplified_directory

    split_directory = config['splitted_directory'] + '/'
    utils.makedir(split_directory)

    for filename in os.listdir(simplified_directory):
        shutil.copy2(simplified_directory + filename, split_directory + filename)
        print('[Data] copied')
        print('  ' + simplified_directory + filename)
        print('  ' + split_directory + filename)