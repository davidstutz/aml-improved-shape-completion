"""
Render for fusion.
"""

import math
import numpy as np
import sys
import os

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
from mesh import Mesh
import pyrender
from scipy import ndimage
from common import Timer
import visualization

def get_points(n_points):
    """
    See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

    :param n_points: number of points
    :type n_points: int
    :return: list of points
    :rtype: numpy.ndarray
    """

    rnd = 1.
    points = []
    offset = 2. / n_points
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(n_points):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_points) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    #visualization.plot_point_cloud(np.array(points))
    return np.array(points)

def get_views(n_views):
    """
    Generate a set of views to generate depth maps from.

    :param n_views: number of views per axis
    :type n_views: int
    :return: rotation matrices
    :rtype: [numpy.ndarray]
    """

    Rs = []
    points = get_points(n_views)

    for i in range(points.shape[0]):
        # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0]**2 + points[i, 1]**2))

        R_x = np.array([[1, 0, 0], [0, math.cos(latitude), -math.sin(latitude)], [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array([[math.cos(longitude), 0, math.sin(longitude)], [0, 1, 0], [-math.sin(longitude), 0, math.cos(longitude)]])

        R = R_y.dot(R_x)
        Rs.append(R)

    return Rs

def render(mesh, Rs):
    """
    Render the given mesh using the generated views.

    :param base_mesh: mesh to render
    :type base_mesh: mesh.Mesh
    :param Rs: rotation matrices
    :type Rs: [numpy.ndarray]
    :return: depth maps
    :rtype: numpy.ndarray
    """

    intrinsics = np.array([
        config['watertight_rendering']['focal_length_x'],
        config['watertight_rendering']['focal_length_y'],
        config['watertight_rendering']['principal_point_x'],
        config['watertight_rendering']['principal_point_x']
    ], dtype=float)
    image_size = np.array([
        config['watertight_rendering']['image_height'],
        config['watertight_rendering']['image_width'],
    ], dtype=np.int32)
    znf = np.array([
        config['watertight_rendering']['mesh_center'][2] - 0.75,
        config['watertight_rendering']['mesh_center'][2] + 0.75
    ], dtype=float)
    depthmaps = []

    for i in range(len(Rs)):
        np_vertices = Rs[i].dot(mesh.vertices.astype(np.float64).T)
        np_vertices[2, :] += config['watertight_rendering']['mesh_center'][2]

        np_faces = mesh.faces.astype(np.float64)
        np_faces += 1

        depthmap, mask, img = pyrender.render(np_vertices.copy(), np_faces.T.copy(), intrinsics, znf, image_size)

        # This is mainly the results of experimentation.
        # We first close holes, and then offset the depth map in order to
        # render the car with more volume.
        # The dilation additionally makes sure that thin structures are
        # preserved.
        depthmap = ndimage.morphology.grey_erosion(depthmap, size=(5, 5))
        depthmap = ndimage.morphology.grey_dilation(depthmap, size=(5, 5))
        depthmap -= config['watertight_rendering']['depth_offset_factor']*config['watertight_fusion']['voxel_size']
        depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))

        depthmaps.append(depthmap)

    return depthmaps

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 2_watertight_render.py config_folder [modulo_base] [modulo_index]')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    modulo_base = 1
    if len(sys.argv) > 2:
        modulo_base = max(1, int(sys.argv[2]))
        print('[Data] modulo base %d' % modulo_base)

    modulo_index = 0
    if len(sys.argv) > 3:
        modulo_index = max(0, int(sys.argv[3]))
        print('[Data] modulo index %d' % modulo_index)

    config_files = [config_file for config_file in os.listdir(config_folder)]
    config = utils.read_json(config_folder + config_files[-1])

    scaled_directory = config['scaled_directory'] + '/'
    assert os.path.exists(scaled_directory), 'directory %s does not exist' % scaled_directory

    depth_directory = config['depth_directory'] + '/'
    utils.makedir(depth_directory)

    off_files = utils.read_ordered_directory(scaled_directory)
    timer = Timer()

    Rs = get_views(config['watertight_rendering']['n_views'])

    for n in range(len(off_files)):
        if (n - modulo_index)%modulo_base == 0:
            timer.reset()
            mesh = Mesh.from_off(off_files[n])
            depths = render(mesh, Rs)

            depth_file = depth_directory + '%d.hdf5' % n
            utils.write_hdf5(depth_file, np.array(depths))
            print('[Data] wrote %s (%f seconds)' % (depth_file, timer.elapsed()))