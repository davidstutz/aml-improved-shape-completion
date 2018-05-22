#!/usr/bin/env python
"""
Simplify.
"""

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import  utils

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 3_1_simplify.py config_folder [modulo_base] [modulo_index]')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder)]
    # On ModelNet, we are given train and testing sets, so everything from
    # 1_scaling to 3_simplifying needs to be done for both.
    # Splitting only needs to be done for the train set.
    config_sets = ['training_inference', 'test']
    config_files = [config_file for config_file in config_files if config_file[:-5] in config_sets]

    modulo_base = 1
    if len(sys.argv) > 2:
        modulo_base = max(1, int(sys.argv[2]))
        print('[Data] modulo base %d' % modulo_base)

    modulo_index = 0
    if len(sys.argv) > 3:
        modulo_index = max(0, int(sys.argv[3]))
        print('[Data] modulo index %d' % modulo_index)

    for config_file in config_files:
        config = utils.read_json(config_folder + config_file)

        watertight_directory = config['watertight_directory'] + '/'
        simplified_directory = config['simplified_directory'] + '/'
        utils.makedir(simplified_directory)

        off_files = utils.read_ordered_directory(watertight_directory)

        for n in range(len(off_files)):
            if (n - modulo_index) % modulo_base == 0:
                os.system('meshlabserver -i %s/%d.off -o %s/%d.off -s %s' % (
                    watertight_directory, n,
                    simplified_directory, n,
                    config['simplification_script']
                ))
