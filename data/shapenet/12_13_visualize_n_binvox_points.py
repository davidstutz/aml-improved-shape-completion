import argparse
import sys
import os
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
from blender_utils import *
import common
import json
import h5py
import numpy as np

def read_json(file):
    """
    Read a JSON file.

    :param file: path to file to read
    :type file: str
    :return: parsed JSON as dict
    :rtype: dict
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        return json.load(fp)

def read_hdf5(file, key = 'tensor'):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param file: path to file to read
    :type file: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray
    """

    assert os.path.exists(file), 'file %s not found' % file

    h5f = h5py.File(file, 'r')
    tensor = h5f[key][()]
    h5f.close()

    return tensor

if __name__ == '__main__':

    try:
        argv = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        log('[Error] "--" not found, call as follows:', LogLevel.ERROR)
        log('[Error] $BLENDER --background --python 12_3_visualize_binvox.py -- 1>/dev/null config_folder', LogLevel.ERROR)
        exit()

    if len(argv) < 1:
        log('[Error] not enough parameters, call as follows:', LogLevel.ERROR)
        log('[Error] $BLENDER --background --python 12_3_visualize_binvox.py -- 1>/dev/null config_folder', LogLevel.ERROR)
        exit()

    config_folder = argv[0] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    num = int(argv[1])
    assert num > 0

    config_files = ['test.json']
    for config_file in config_files:
        config = read_json(config_folder + config_file)

        height = config['height']
        width = config['width']
        depth = config['depth']
        scale = 1./max(height, width, depth)

        n_observations = config['n_observations']

        inputs = read_hdf5(common.filename(config, 'input_file', '.h5', num))
        inputs = inputs.reshape((inputs.shape[0]*n_observations, 1, inputs.shape[2], inputs.shape[3], inputs.shape[4]))

        vis_directory = common.dirname(config, 'vis_dir')
        if not os.path.isdir(vis_directory):
            os.makedirs(vis_directory)

        voxel_size = 0.007
        if height >= 32:
            voxel_size = 0.0055
        if height >= 48:
            voxel_size = 0.004
        log('[Data] voxel size ' + str(voxel_size))

        N = 30
        log('[Data] %d samples' % inputs.shape[0])
        for i in range(N):
            n = i * (inputs.shape[0] // N)

            camera_target = initialize()
            input_material = make_material('BRC_Material_Point_Cloud', (0.65, 0.23, 0.25), 1, True)

            load_volume(inputs[n][0], voxel_size, input_material, (0, 0, 0), (width*scale, depth*scale, height*scale), 'zxy')

            rotation = (5, 0, -55)
            distance = 0.35
            png_file = vis_directory + '/%d_%d_bin_points.png' % (n, num)
            render(camera_target, png_file, rotation, distance)

            log('[Data] wrote %s' % png_file)