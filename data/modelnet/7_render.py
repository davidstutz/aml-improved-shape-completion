"""
Render.
"""

import numpy as np
import math
import pyrender

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import  utils
from mesh import Mesh
import common

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 7_render.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder) if not (config_file.find('prior') > 0)]
    for config_file in config_files:
        config = utils.read_json(config_folder + config_file)

        image_width = config['image_width']
        image_height = config['image_height']

        intrinsics = np.array([config['focal_length_x'], config['focal_length_y'], config['principal_point_x'], config['principal_point_y']], dtype = np.float64)
        size = np.array([image_height, image_width], dtype = np.int32)
        mesh_center = (config['mesh_center_x'], config['mesh_center_y'], config['mesh_center_z'])
        znf = np.array([config['z_near'], config['z_far']], dtype=float)

        height = config['height']
        width = config['width']
        depth = config['depth']
        scale = max(height, width, depth)

        suffix = config['suffix']

        off_directory = config['off_dir'] + '/'
        depth_file = common.filename(config, 'depth_file')
        angles_file = common.filename(config, 'render_orientation_file')

        off_files = utils.read_ordered_directory(off_directory)
        n_files = len(off_files)
        print('[Data] found %d off files' % n_files)

        n_observations = config['n_observations']
        depth_maps = np.zeros((n_files, n_observations, image_height, image_width))
        rotations = np.zeros((n_files, n_observations, 3))

        for n in range(n_files):
            base_mesh = Mesh.from_off(off_files[n])

            for k in range(n_observations):
                mesh = base_mesh.copy()
                rotations[n, k, 0] = (np.random.random()*(config['render_max_x_rotation'] + abs(config['render_min_x_rotation'])) - abs(config['render_min_x_rotation'])) / 180.*math.pi
                rotations[n, k, 1] = (np.random.random()*(config['render_max_y_rotation'] + abs(config['render_min_y_rotation'])) - abs(config['render_min_y_rotation'])) / 180.*math.pi

                mesh.rotate(rotations[n, k])
                mesh.translate(mesh_center)

                np_vertices = mesh.vertices.astype(np.float64)
                np_faces = mesh.faces.astype(np.float64)
                np_faces += 1

                depth_map, mask, img = pyrender.render(np_vertices.T.copy(), np_faces.T.copy(), intrinsics, znf, size)

                # A main issue of rendering and subsequent voxelization is that (due to
                # the discretiation) observed voxels may lie BEHIND the actual surface (after voxelization).
                # There we offset the depthmap by a small constant.
                #depth_map -= 1/(2*scale)

                depth_maps[n][k] = depth_map

            print('[Data] rendered %s %d/%d' % (off_files[n], (n + 1), n_files))

        utils.write_hdf5(angles_file, rotations)
        print('[Data] wrote %s' % angles_file)
        utils.write_hdf5(depth_file, depth_maps)
        print('[Data] wrote %s' % depth_file)
