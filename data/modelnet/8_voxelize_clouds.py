"""
Voxelize clouds.
"""

import os
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 8_voxelize_clouds.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_file = ''
    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    if config_file and os.path.exists(config_folder + config_file):
        print('[Data] processing %s only' % config_folder + config_file)
        os.system('./libvoxelizecloud/bin/voxelize_clouds ' + config_folder + config_file)
    else:
        print('[Data] processing all files in %s' % config_folder)
        config_files = [config_file for config_file in os.listdir(config_folder) if not (config_file.find('prior') > 0)]
        for config_file in config_files:
            os.system('./libvoxelizecloud/bin/voxelize_clouds ' + config_folder + config_file)