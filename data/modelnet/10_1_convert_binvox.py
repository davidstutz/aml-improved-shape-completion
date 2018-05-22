"""
Reconstruct.
"""

import os
import sys

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
import mcubes
import binvox_rw

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 10_reconstruct.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    for config_file in os.listdir(config_folder):
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)

        filled_file = common.filename(config, 'filled_file')
        assert os.path.exists(filled_file), 'file %s does not exist' % filled_file

        filled = utils.read_hdf5(filled_file)
        filled = filled.squeeze()
        
        binvox_directory = common.dirname(config, 'binvox_dir')
        utils.makedir(binvox_directory)

        for n in range(filled.shape[0]):
            model = binvox_rw.Voxels(filled[n] > 0.5, filled[n].shape, (0, 0, 0), 1)
            binvox_file = binvox_directory + str(n) + '.binvox'
            with open(binvox_file, 'w') as fp:
                model.write(fp)
                print('[Validation] wrote ' + binvox_file)