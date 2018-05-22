"""
Reconstruct.
"""

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
from point_cloud import PointCloud
import numpy as np
import visualization

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 13_ply_observations.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_file = config_folder + 'training_prior.json'
    assert os.path.exists(config_file)

    print('[Data] reading ' + config_file)
    config = utils.read_json(config_file)

    filled_file = common.filename(config, 'filled_file')
    filled = utils.read_hdf5(filled_file)
    print('[Data] read ' + filled_file)

    slices = []
    for j in range(2, filled[0][0].shape[0] - 2, 2):
        slices.append(j)

    N = filled.shape[0]
    statistics = np.sum(np.squeeze(filled), axis=0)/float(N)
    statistics = 1 - statistics
    statistics = (0.75*statistics + 0.25*1)**2

    visualization.plot_specific_slices([statistics], slices)

    statistics_file = common.filename(config, 'training_statistics_file')
    utils.write_hdf5(statistics_file, statistics)
    print('[Data] wrote ' + statistics_file)