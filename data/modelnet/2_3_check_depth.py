"""
Reconstruct.
"""

import os
import sys
from matplotlib import pyplot

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 2_3_check_depth.py config_folder [index]')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder)]
    config = utils.read_json(config_folder + config_files[-1])

    index = -1
    if len(sys.argv) > 2:
        index = int(sys.argv[2])

    if index >= 0:
        depths = utils.read_hdf5(config['depth_directory'] + '/%d.hdf5' % index)

        for i in range(depths.shape[0]):
            pyplot.clf()
            pyplot.imshow(depths[i], interpolation='none')
            pyplot.savefig('%d_%d_depth.png' % (index, i))
            print('[Data] wrote %d_%d_depth.png' % (index, i))
    else:
        n = 0
        depth_files = utils.read_ordered_directory(config['depth_directory'])
        for depth_file in depth_files:
            depths = utils.read_hdf5(depth_file)

            for i in range(depths.shape[0]):
                pyplot.clf()
                pyplot.imshow(depths[i], interpolation='none')
                pyplot.savefig('%d_%d_depth.png' % (n, i))
                print('[Data] wrote %d_%d_depth.png' % (n, i))

            n += 1
