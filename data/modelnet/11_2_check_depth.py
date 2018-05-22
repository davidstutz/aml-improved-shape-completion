"""
Reconstruct.
"""

import os
import sys
from matplotlib import pyplot

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 10_reconstruct.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder) if not (config_file.find('prior') > 0)]
    for config_file in config_files:
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)
        set = config_file[:-5]

        depths = utils.read_hdf5(common.filename(config, 'depth_file'))
        print('[Data] read ' + common.filename(config, 'depth_file'))
        print('[Data] depths: %s' % ' x '.join(map(str, depths.shape)))


        for i in range(min(25, depths.shape[0])):
            n = i
            print('[Data] visualizing %s %d/%d' % (set, (n + 1), depths.shape[0]))

            pyplot.clf()
            pyplot.imshow(depths[n][0], interpolation='none')
            pyplot.savefig('%s_%d_depth.png' % (set, n))
