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

def filename(config, key, num):
    return config[key] + '_' + str(config['multiplier']) + '_' \
           + str(config['image_height']) + 'x' + str(config['image_width']) + '_' \
           + str(config['height']) + 'x' + str(config['width']) + 'x' + str(config['depth']) \
           + '_' + str(num) + config['suffix'] + '.h5'

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

        outputs = utils.read_hdf5(common.filename(config, 'filled_file'))
        print('[Data] read ' + common.filename(config, 'filled_file'))

        num = 2
        inputs = utils.read_hdf5(filename(config, 'input_file', num))
        print('[Data] read ' + filename(config, 'input_file', num))

        space = utils.read_hdf5(filename(config, 'space_file', num))
        print('[Data] read ' + filename(config, 'space_file', num))

        print('[Data] outputs: %s' % ' x '.join(map(str, outputs.shape)))
        print('[Data] inputs: %s' % ' x '.join(map(str, inputs.shape)))
        print('[Data] space: %s' % ' x '.join(map(str, space.shape)))

        n_observations = config['n_observations']

        for i in range(min(25, outputs.shape[0])):
            #n = random.randint(0, outputs[-1].shape[0])
            n = i
            print('[Data] visualizing %s %d/%d' % (set, (n + 1), outputs.shape[0]))

            for k in range(n_observations):

                slices = []
                for j in range(2, outputs[n][0].shape[0] - 2, 2):
                    slices.append(j)

                outputs_inputs = outputs[n][0].copy()
                outputs_inputs[inputs[n][k] == 1] = 2
                space_inputs = space[n][k].copy()
                space_inputs[inputs[n][k] == 1] = 2
                mask = np.logical_and(inputs[n][k] == 1, space[n][k] == 1)
                space_inputs[mask] = 3

                visualization.plot_specific_slices([outputs[n][0], inputs[n][k], space[n][k], outputs_inputs, space_inputs], slices, '%s_%d_%d.png' % (set, n, k), 0, 0, 3)
