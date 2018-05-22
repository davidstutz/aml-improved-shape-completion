"""
Voxelize meshs.
"""

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
import shutil

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('[Data] Usage python 6_b_copy.py config_folder_from config_folder_to')
        exit(1)

    from_config_folder = sys.argv[1] + '/'
    to_config_folder = sys.argv[2] + '/'

    assert os.path.exists(from_config_folder)
    assert os.path.exists(to_config_folder)

    print('[Data] copying')
    print('  off_gt_dir')
    print('  sdf_file')
    print('  output_file')
    print('  filled_file')

    hdf5_file_keys = [
        'sdf_file',
        'output_file',
        'filled_file',
    ]

    dir_keys = [
        'off_gt_dir'
    ]

    config_files = [config_file for config_file in os.listdir(from_config_folder)]
    for config_file in config_files:
        assert os.path.exists(to_config_folder + config_file)

    for config_file in config_files:
        from_config = utils.read_json(from_config_folder + config_file)
        to_config = utils.read_json(to_config_folder + config_file)

        for key in hdf5_file_keys:
            from_file = common.filename(from_config, key)
            to_file = common.filename(to_config, key)

            print('[Data] copying')
            print('  ' + from_file)
            print('  ' + to_file)
            shutil.copy(from_file, to_file)

        for key in dir_keys:
            from_dir = common.dirname(from_config, key)
            to_dir = common.dirname(to_config, key)

            # !
            assert not os.path.exists(to_dir)

            print('[Data] copying')
            print('  ' + from_dir)
            print('  ' + to_dir)
            shutil.copytree(from_dir, to_dir)