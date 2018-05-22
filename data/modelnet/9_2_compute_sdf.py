"""
Post-process.
"""

import numpy as np
import os
import sys

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 1_post_process.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder)]
    for config_file in config_files:
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)

        if config['sdf']:
            truncation = config['truncation']
            sdfs = utils.read_hdf5(common.filename(config, 'sdf_file'))

            tsdfs = sdfs.copy()
            tsdfs[tsdfs > truncation] = truncation
            tsdfs[tsdfs < -truncation] = -truncation

            ltsdfs = tsdfs.copy()
            ltsdfs[ltsdfs > 0] = np.log(ltsdfs[ltsdfs > 0] + 1)
            ltsdfs[ltsdfs < 0] = - np.log(np.abs(ltsdfs[ltsdfs < 0]) + 1)

            tsdf_file = common.filename(config, 'tsdf_file')
            ltsdf_file = common.filename(config, 'ltsdf_file')

            utils.write_hdf5(tsdf_file, tsdfs)
            print('[Data] wrote ' + tsdf_file)
            utils.write_hdf5(ltsdf_file, ltsdfs)
            print('[Data] wrote ' + ltsdf_file)

    config_files = [config_file for config_file in os.listdir(config_folder) if config_file.find('prior') < 0]
    for config_file in config_files:
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)

        if config['sdf']:
            input_sdfs = utils.read_hdf5(common.filename(config, 'input_sdf_file'))

            input_tsdfs = input_sdfs.copy()
            input_tsdfs[input_tsdfs > truncation] = truncation
            input_tsdfs[input_tsdfs < -truncation] = -truncation

            input_ltsdfs = input_tsdfs.copy()
            input_ltsdfs[input_ltsdfs > 0] = np.log(input_ltsdfs[input_ltsdfs > 0] + 1)
            input_ltsdfs[input_ltsdfs < 0] = - np.log(np.abs(input_ltsdfs[input_ltsdfs < 0]) + 1)

            input_tsdf_file = common.filename(config, 'input_tsdf_file')
            input_ltsdf_file = common.filename(config, 'input_ltsdf_file')

            utils.write_hdf5(input_tsdf_file, input_tsdfs)
            print('[Data] wrote ' + input_tsdf_file)
            utils.write_hdf5(input_ltsdf_file, input_ltsdfs)
            print('[Data] wrote ' + input_ltsdf_file)
