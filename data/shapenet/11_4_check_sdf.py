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

        input_sdfs = utils.read_hdf5(common.filename(config, 'input_sdf_file'))
        print('[Data] read ' + common.filename(config, 'input_sdf_file'))
        input_ltsdfs = utils.read_hdf5(common.filename(config, 'input_ltsdf_file'))
        print('[Data] read ' + common.filename(config, 'input_ltsdf_file'))

        output_sdfs = utils.read_hdf5(common.filename(config, 'sdf_file'))
        print('[Data] reading ' + common.filename(config, 'sdf_file'))
        output_ltsdfs = utils.read_hdf5(common.filename(config, 'ltsdf_file'))
        print('[Data] reading ' + common.filename(config, 'ltsdf_file'))

        print('[Data] input_sdfs: %s' % ' x '.join(map(str, input_sdfs.shape)))
        print('[Data] input_ltsdfs: %s' % ' x '.join(map(str, input_ltsdfs.shape)))
        print('[Data] output_sdfs: %s' % ' x '.join(map(str, output_sdfs.shape)))
        print('[Data] output_ltsdfs: %s' % ' x '.join(map(str, output_ltsdfs.shape)))

        for i in range(min(25, output_sdfs.shape[0])):
            #n = random.randint(0, filled[-1].shape[0])
            n = i
            print('[Data] visualizing %s %d/%d' % (set, (n + 1), output_sdfs.shape[0]))

            slices = []
            for j in range(2, output_sdfs[n][0].shape[0] - 2, 2):
                slices.append(j)

            visualization.plot_specific_slices([
                output_sdfs[n][0],
                output_ltsdfs[n][0],
                input_sdfs[n][0],
                input_ltsdfs[n][0]
            ], slices, set + '_' + str(n) + '.png', 0, np.min(output_sdfs[n][0]), max(np.max(output_sdfs[n][0]), np.max(input_sdfs[n][0])))
