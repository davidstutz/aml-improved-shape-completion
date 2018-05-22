"""
Reconstruct.
"""

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
from point_cloud import PointCloud

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 13_ply_observations.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder) if not (config_file.find('prior') > 0)]
    for config_file in config_files:
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)

        for k in range(config['n_observations']):
            txt_directory = common.dirname(config, 'txt_gt_dir') + str(k) + '/'
            assert os.path.exists(txt_directory)

            ply_directory = common.dirname(config, 'ply_gt_dir') + str(k) + '/'
            if not os.path.exists(ply_directory):
                os.makedirs(ply_directory)

            for filename in os.listdir(txt_directory):
                point_cloud = PointCloud.from_txt(txt_directory + filename)
                point_cloud.to_ply(ply_directory + filename[:-4] + '.ply')
                print('[Data] wrote ' + ply_directory + filename[:-4] + '.ply')