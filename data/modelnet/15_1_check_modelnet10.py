"""
Reconstruct.
"""

import os
import sys
import numpy as np

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
import visualization

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

        filled = utils.read_hdf5(common.filename(config, 'filled_file'))
        print('[Data] read ' + common.filename(config, 'filled_file'))

        inputs = utils.read_hdf5(common.filename(config, 'input_file'))
        print('[Data] read ' + common.filename(config, 'input_file'))

        space = utils.read_hdf5(common.filename(config, 'space_file'))
        print('[Data] read ' + common.filename(config, 'space_file'))

        print('[Data] filled: %s' % ' x '.join(map(str, filled.shape)))
        print('[Data] inputs: %s' % ' x '.join(map(str, inputs.shape)))
        print('[Data] space: %s' % ' x '.join(map(str, space.shape)))

        sdf = ('sdf' in config.keys() and config['sdf']) or ('synthetic_sdf' in config.keys() and config['synthetic_sdf'])
        if sdf:
            sdfs = utils.read_hdf5(common.filename(config, 'sdf_file'))
            print('[Data] reading ' + common.filename(config, 'sdf_file'))
            print('[Data] sdfs: %s' % ' x '.join(map(str, sdfs.shape)))

        for i in range(10):
            n = i*config['limit']*config['multiplier']
            print('[Data] visualizing %s %d/%d' % (set, (n + 1), filled.shape[0]))

            slices = []
            for j in range(2, filled[n][0].shape[0] - 2, 2):
                slices.append(j)

            filled_inputs = filled[n][0].copy()
            filled_inputs[inputs[n][0] == 1] = 2
            space_inputs = space[n][0].copy()
            space_inputs[inputs[n][0] == 1] = 2
            mask = np.logical_and(inputs[n][0] == 1, space[n][0] == 1)
            space_inputs[mask] = 3

            if sdf:
                sdf_bin = sdfs[n][0].copy()
                sdf_bin[sdf_bin >= 0] = 0
                sdf_bin[sdf_bin < 0] = 1
                visualization.plot_specific_slices([filled[n][0], inputs[n][0], space[n][0], filled_inputs, space_inputs, sdf_bin], slices, set + '_' + str(n) + '.png', 0, 0, 3)
            else:
                visualization.plot_specific_slices([filled[n][0], inputs[n][0], space[n][0], filled_inputs, space_inputs], slices, set + '_' + str(n) + '.png', 0, 0, 3)