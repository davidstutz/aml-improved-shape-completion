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

        if config['sdf']:
            sdfs = utils.read_hdf5(common.filename(config, 'sdf_file'))
            print('[Data] reading ' + common.filename(config, 'sdf_file'))
            print('[Data] sdfs: %s' % ' x '.join(map(str, sdfs.shape)))

        n_observations = config['n_observations']

        for i in range(min(25, filled.shape[0])):
            #n = random.randint(0, filled[-1].shape[0])
            n = i
            print('[Data] visualizing %s %d/%d' % (set, (n + 1), filled.shape[0]))

            for k in range(n_observations):

                slices = []
                for j in range(2, filled[n][0].shape[0] - 2, 2):
                    slices.append(j)

                filled_inputs = filled[n][0].copy()
                filled_inputs[inputs[n][k] == 1] = 2
                space_inputs = space[n][k].copy()
                space_inputs[inputs[n][k] == 1] = 2
                mask = np.logical_and(inputs[n][k] == 1, space[n][k] == 1)
                space_inputs[mask] = 3

                if config['sdf']:
                    sdf_bin = sdfs[n][0].copy()
                    sdf_bin[sdf_bin >= 0] = 0
                    sdf_bin[sdf_bin < 0] = 1
                    visualization.plot_specific_slices([filled[n][0], inputs[n][k], space[n][k], filled_inputs, space_inputs, sdf_bin], slices, '%s_%d_%d.png' % (set, n, k), 0, 0, 3)
                else:
                    visualization.plot_specific_slices([filled[n][0], inputs[n][k], space[n][k], filled_inputs, space_inputs], slices, '%s_%d_%d.png' % (set, n, k), 0, 0, 3)
