"""
Split.
"""

import random
import shutil

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import ntpath

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 4_split.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    taken = []
    files = []
    numbers = {}

    for config_file in os.listdir(config_folder):
        config = utils.read_json(config_folder + config_file)
        simplified_directory = config['simplified_directory'] + '/'
        assert os.path.exists(simplified_directory), 'directory %s does not exist' % simplified_directory

        split_directory = config['splitted_directory'] + '/'
        set = config_file[:-5]
        set_directory = split_directory + set + '/'
        utils.makedir(set_directory)

        off_file = split_directory + set + '.txt'
        with open(off_file, 'r') as f:
            files = f.readlines();
            files = [filepath.strip() for filepath in files]

        i = 0
        for filepath in files:
            off_from_file = filepath
            off_to_file = set_directory + '%d.off' % i
            shutil.copy(off_from_file, off_to_file)
            print('[Data] copied')
            print('   %s' % off_from_file)
            print('   %s' % off_to_file)

            i += 1