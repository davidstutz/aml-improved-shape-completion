"""
Post-process.
"""

import numpy as np
import os
import sys

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 1_post_process.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    for config_file in os.listdir(config_folder):
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)

        # First we make a minor fix for the space.
        # We do not allow voxels marked as free space which are also occupied,
        # but during voxelization this usually happens!

        space_file = common.filename(config, 'space_file')
        if os.path.exists(space_file):
            space = utils.read_hdf5(space_file)
            input_file = common.filename(config, 'input_file')
            input = utils.read_hdf5(input_file)

            space[input == 1] = 0
            if len(space.shape) < 5:
                space = np.expand_dims(space, axis=1)

            utils.write_hdf5(space_file, space)
            print('[Data] wrote ' + space_file)

        keys = ['input', 'space', 'output', 'filled', 'sdf', 'tsdf', 'ltsdf', 'input_sdf', 'input_tsdf', 'input_ltsdf']

        for key in keys:
            file = common.filename(config, key + '_file')
            if not os.path.exists(file):
                print('[Data] not processing ' + file)
                continue

            print('[Data] processing ' + file)
            assert os.path.exists(file), 'file %s not found' % file

            volumes = utils.read_hdf5(file)
            volumes = np.squeeze(volumes)

            if len(volumes.shape) < 4:
                volumes = np.expand_dims(volumes, axis=0)

            if len(volumes.shape) < 5:
                utils.write_hdf5(file, np.expand_dims(volumes, axis=1))
                print('[Data] wrote ' + file)
            else:
                print('[Data] not needed to process ' + file)
