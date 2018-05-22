"""
Split.
"""

import random
import shutil

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common

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
        utils.makedir(split_directory)
        set = config_file[:-5]

        numbers[set] = []

        # get all files only once to guarantee the same order
        if len(files) == 0:
            files = utils.read_ordered_directory(simplified_directory)
            taken = [False] * len(files)

        print('[Data] choosing %s' % set)
        n = config['number']

        for i in range(n):
            j = 0
            k = 0
            while taken[j] and k < 1e6: # prevent infinite loop
                j = random.randint(0, len(files) - 1)
                k = k + 1

            taken[j] = True
            numbers[set].append(files[j])

            # try to read the file to see if OFF files are valid
            off_from_file = files[j]
            utils.read_off(off_from_file)

        off_file = split_directory + set + '.txt'
        with open(off_file, 'w') as f:
            for j in numbers[set]:
                f.write(str(j) + '\n')
            print('[Data] wrote' + off_file)