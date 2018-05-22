"""
Fusion.
"""

import math
import numpy as np
import sys
import os

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
from common import Timer
import mcubes
import time

# For getting views.
watertight_render = __import__('2_1_watertight_render')

# Whether to use GPU or CPU for depth fusion.
use_gpu = True

if use_gpu:
    import libfusiongpu as libfusion
    from libfusiongpu import tsdf_gpu as compute_tsdf

else:
    import libfusioncpu as libfusion
    from libfusioncpu import tsdf_cpu as compute_tsdf

def fusion(depthmaps, Rs):
    """
    Fuse the rendered depth maps.

    :param depthmaps: depth maps
    :type depthmaps: numpy.ndarray
    :param Rs: rotation matrices corresponding to views
    :type Rs: [numpy.ndarray]
    :return: (T)SDF
    :rtype: numpy.ndarray
    """

    Ks = np.array([
        [config['watertight_rendering']['focal_length_x'], 0, config['watertight_rendering']['principal_point_x']],
        [0, config['watertight_rendering']['focal_length_y'], config['watertight_rendering']['principal_point_y']],
        [0, 0, 1]
    ])
    Ks = Ks.reshape((1, 3, 3))
    Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

    Ts = []
    for i in range(len(Rs)):
        Rs[i] = Rs[i]
        Ts.append(np.array(config['watertight_rendering']['mesh_center']))

    Ts = np.array(Ts).astype(np.float32)
    Rs = np.array(Rs).astype(np.float32)

    depthmaps = np.array(depthmaps).astype(np.float32)
    views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)

    # Note that this is an alias defined as libfusiongpu.tsdf_gpu or libfusioncpu.tsdf_cpu!
    return compute_tsdf(views, config['watertight_fusion']['resolution'], config['watertight_fusion']['resolution'],
                        config['watertight_fusion']['resolution'], config['watertight_fusion']['voxel_size'], config['watertight_fusion']['truncation'], False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 2_2_watertight_fuse.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder)]

    # On ModelNet, we are given train and testing sets, so everything from
    # 1_scaling to 3_simplifying needs to be done for both.
    # Splitting only needs to be done for the train set.
    config_sets = ['training_inference', 'test']
    config_files = [config_file for config_file in config_files if config_file[:-5] in config_sets]

    modulo_base = 1
    if len(sys.argv) > 2:
        modulo_base = max(1, int(sys.argv[2]))
        print('[Data] modulo base %d' % modulo_base)

    modulo_index = 0
    if len(sys.argv) > 3:
        modulo_index = max(0, int(sys.argv[3]))
        print('[Data] modulo index %d' % modulo_index)

    for config_file in config_files:
        config = utils.read_json(config_folder + config_file)

        scaled_directory = config['scaled_directory'] + '/'
        assert os.path.exists(scaled_directory), 'directory %s does not exist' % scaled_directory

        scaled_files = utils.read_ordered_directory(scaled_directory)
        n_files_expected = len(scaled_files)

        depth_directory = config['depth_directory'] + '/'
        while not os.path.exists(depth_directory):
            print('[Data] waiting for %s' % depth_directory)
            time.sleep(10)

        watertight_directory = config['watertight_directory'] + '/'
        tsdf_directory = config['tsdf_directory'] + '/'

        utils.makedir(watertight_directory)
        utils.makedir(tsdf_directory)
        timer = Timer()

        Rs = watertight_render.get_views(config['watertight_rendering']['n_views'])

        for n in range(n_files_expected):
            if (n - modulo_index) % modulo_base == 0:
                tsdf_file = tsdf_directory + '/%d.hdf5' % n
                off_file = watertight_directory + '/%d.off' % n

                # As rendering might be slower, we wait for rendering to finish.
                # This allows to run rendering and fusing in parallel (more or less).

                waited = False
                depth_file = depth_directory + '%d.hdf5' % n

                while not os.path.exists(depth_file):
                    waited = True
                    print('[Data] waiting for %s' % depth_file)
                    time.sleep(10)

                # Wait for synchronization.
                if waited:
                    time.sleep(10)

                try:
                    # Sometimes signature of HDF5 files is still not available.
                    depths = utils.read_hdf5(depth_file)
                except IOError:
                    print('[Data] could not read %s' % depth_file)
                    time.sleep(5)

                # Try again, now it can really fail if file is not there.
                depths = utils.read_hdf5(depth_file)

                timer.reset()
                tsdf = fusion(depths, Rs)
                tsdf = tsdf[0]

                utils.write_hdf5(tsdf_file, tsdf)
                print('[Data] wrote %s (%f seconds)' % (tsdf_file, timer.elapsed()))

                vertices, triangles = mcubes.marching_cubes(-tsdf, 0)
                vertices /= config['watertight_fusion']['resolution']
                vertices -= 0.5
                mcubes.export_off(vertices, triangles, off_file)
                print('[Data] wrote %s (%f seconds)' % (off_file, timer.elapsed()))