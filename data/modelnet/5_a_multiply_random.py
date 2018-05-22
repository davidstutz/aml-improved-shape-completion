"""
Multiply.
"""

import math
import numpy as np
from scipy import ndimage

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import  utils
from mesh import Mesh
import common

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 5_multiply.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    for config_file in os.listdir(config_folder):
        config = utils.read_json(config_folder + config_file)
        set = config_file[:-5]

        base_directory = config['splitted_directory'] + '/'
        assert os.path.exists(base_directory), 'directory %s does not exist' % base_directory

        off_directory = config['off_dir'] + '/'
        utils.makedir(off_directory)

        i = 0
        off_files = utils.read_ordered_directory(base_directory)

        for n in range(len(off_files)):
            mesh = Mesh.from_off(off_files[n])

            for j in range(config['multiplier']):
                mesh_prime = mesh.copy()

                rotation = [
                    (np.random.random()*2*config['max_x_rotation'] - config['max_x_rotation'])/180.0*math.pi,
                    (np.random.random()*2*config['max_y_rotation'] - config['max_y_rotation'])/180.0*math.pi,
                    (np.random.random()*2*config['max_z_rotation'] - config['max_z_rotation'])/180.0*math.pi
                ]
                mesh_prime.rotate(rotation)

                scale = 1 + np.random.uniform(config['min_scale'], config['max_scale'])
                scales = [scale, scale, scale]
                mesh_prime.scale(scales)

                translation = [
                    np.random.uniform(config['min_x_translation'], config['max_x_translation']),
                    np.random.uniform(config['min_y_translation'], config['max_y_translation']),
                    np.random.uniform(config['min_z_translation'], config['max_z_translation'])
                ]
                mesh_prime.translate(translation)

                out_file = off_directory + '/%d.off' % i
                mesh_prime.to_off(out_file)

                print('[Data] generated %s' % out_file)
                i += 1