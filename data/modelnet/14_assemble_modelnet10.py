"""
Reconstruct.
"""

import os
import sys
import numpy as np

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
import ntpath
import shutil

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 13_assemble_modelnet10.py config_folder')
        exit(1)

    to_config_folder = sys.argv[1] + '/'
    assert os.path.exists(to_config_folder)

    type = 'clean'
    resolution = sys.argv[2]
    config_folders = [
        'bathtub',
        'bed',
        'chair',
        'desk',
        'dresser',
        'monitor',
        'night_stand',
        'sofa',
        'table',
        'toilet',
    ]

    for i in range(len(config_folders)):
        config_folders[i] = 'config/' + config_folders[i] + '.' + type + '.' + resolution + '/'

    # 1 Assemble all HDF5 files.
    hdf5_file_keys = [
        'sdf_file',
        'output_file',
        'filled_file',
        'tsdf_file',
        'ltsdf_file',
    ]

    # 2 Assemble all directories (mostly OFF and TXT files.
    dir_keys = [
        'off_gt_dir',
    ]

    config_files = [config_file for config_file in os.listdir(to_config_folder)]
    for config_file in config_files:
        for config_folder in config_folders:
            assert os.path.exists(config_folder + config_file), 'file %s not found' % (config_folder + config_file)
            print('[Data] found ' + config_folder + config_file)

    for config_file in config_files:
        to_config = utils.read_json(to_config_folder + config_file)

        for key in hdf5_file_keys:
            data = []

            print('[Data] +')
            for from_config_folder in config_folders:
                from_config = utils.read_json(from_config_folder + config_file)

                print('[Data] | reading ' + common.filename(from_config, key))
                data.append(utils.read_hdf5(common.filename(from_config, key)))

            print('[Data] \'-> writing ' + common.filename(to_config, key))
            data = np.concatenate(tuple(data), axis=0)
            utils.write_hdf5(common.filename(to_config, key), data)

        key = 'off_gt_dir'
        count = 0

        for key in dir_keys:
            for from_config_folder in config_folders:
                from_config = utils.read_json(from_config_folder + config_file)

                from_dir = common.dirname(from_config, key)
                to_dir = common.dirname(to_config, key)
                utils.makedir(to_dir)

                files = utils.read_ordered_directory(from_dir)
                for i in range(len(files)):
                    from_file = files[i]
                    to_file = to_dir + '/%d.%s' % (count, from_file[-3:])

                    if i == 0:
                        print('[Data] +')
                        print('[Data] |- ' + from_file)
                        print('[Data] \'-> ' + to_file)
                    shutil.copy(from_file, to_file)
                    count += 1

    # 1 Assemble all HDF5 files.
    hdf5_file_keys = [
        'depth_file',
        'render_orientation_file',
        'input_file',
        'input_sdf_file',
        'space_file',
        'input_tsdf_file',
        'input_ltsdf_file',
    ]

    # 2 Assemble all directories (mostly OFF and TXT files.
    dir_keys = [
        'txt_gt_dir',
    ]

    for config_file in config_files:
        if config_file.find('training_prior') >= 0:
            continue;

        to_config = utils.read_json(to_config_folder + config_file)

        for key in hdf5_file_keys:
            data = []

            print('[Data] +')
            for from_config_folder in config_folders:
                from_config = utils.read_json(from_config_folder + config_file)

                print('[Data] | reading ' + common.filename(from_config, key))
                data.append(utils.read_hdf5(common.filename(from_config, key)))

            print('[Data] \'-> writing ' + common.filename(to_config, key))
            data = np.concatenate(tuple(data), axis=0)
            utils.write_hdf5(common.filename(to_config, key), data)


        n_observations = to_config['n_observations']

        for key in dir_keys:
            counts = [0] * n_observations
            for from_config_folder in config_folders:
                from_config = utils.read_json(from_config_folder + config_file)

                from_dir = common.dirname(from_config, key)
                to_dir = common.dirname(to_config, key)

                for k in range(n_observations):
                    from_dir_k = from_dir + '%d/' % k
                    to_dir_k = to_dir + '%d/' % k
                    utils.makedir(to_dir_k)

                    files = utils.read_ordered_directory(from_dir_k)
                    for i in range(len(files)):
                        from_file = files[i]
                        to_file = to_dir_k + '/%d.%s' % (counts[k], from_file[-3:])

                        if i == 0:
                            print('[Data] +')
                            print('[Data] |- ' + from_file)
                            print('[Data] \'-> ' + to_file)
                        shutil.copy(from_file, to_file)
                        counts[k] += 1
