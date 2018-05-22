"""
Voxelize meshs.
"""

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import glob
import ntpath

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 6_voxelize_meshs.py config_folder [config_file]')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_file = ''
    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    if config_file and os.path.exists(config_folder + config_file):
        print('[Data] processing %s only' % config_folder + config_file)
        os.system('./libvoxelizemesh/bin/voxelize_meshs ' + config_folder + config_file)
    else:
        print('[Data] processing all files in %s' % config_folder)
        for config_file in glob.glob(config_folder + '/*.json'):
            config = utils.read_json(config_file)
            os.system('./libvoxelizemesh/bin/voxelize_meshs ' + config_file)
