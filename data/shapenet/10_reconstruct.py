"""
Reconstruct.
"""

import os
import sys

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
import mcubes

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 10_reconstruct.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    for config_file in os.listdir(config_folder):
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)

        sdf_file = common.filename(config, 'sdf_file')
        assert os.path.exists(sdf_file), 'file %s does not exist' % sdf_file

        sdfs = utils.read_hdf5(sdf_file)
        sdfs = sdfs.squeeze()
        
        reconstructed_directory = common.dirname(config, 'reconstructed_dir')
        utils.makedir(reconstructed_directory)

        for n in range(sdfs.shape[0]):
            vertices, triangles = mcubes.marching_cubes(-sdfs[n].transpose(1, 0, 2), 0)

            off_file = reconstructed_directory + '/%d.off' % n
            mcubes.export_off(vertices, triangles, off_file)
            print('[Data] wrote %s' % off_file)