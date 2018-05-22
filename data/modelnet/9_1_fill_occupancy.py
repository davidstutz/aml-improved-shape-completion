"""
Post-process.
"""

import numpy as np
import os
import sys

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
from skimage import morphology

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 1_post_process.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    for config_file in os.listdir(config_folder):
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)

        outputs = utils.read_hdf5(common.filename(config, 'output_file'))
        outputs = np.squeeze(outputs)
        print('[Data] read ' + common.filename(config, 'output_file'))

        filled = np.zeros(outputs.shape)

        for n in range(outputs.shape[0]):
            labels, num_labels = morphology.label(outputs[n], background=1, connectivity=1, return_num=True)
            outside_label = labels[0][0][0]

            filled[n][labels != outside_label] = 1
            #filled[n][labels == outside_label] = 0

            print('[Data] %s, filling %d/%d (%d labels)' % (config_file[:-5], (n + 1), outputs.shape[0], num_labels))

        utils.write_hdf5(common.filename(config, 'filled_file'), filled)
        print('[Data] wrote ' + common.filename(config, 'filled_file'))
