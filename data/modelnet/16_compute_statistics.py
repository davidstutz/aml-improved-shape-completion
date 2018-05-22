"""
Visualize results of an experiment.
"""

import numpy as np
import os
import terminaltables
import scipy.ndimage.morphology as morph
from scipy import ndimage
from scipy import misc
import glob
import pickle
import math
import matplotlib
from matplotlib import pyplot as plt
import shutil

import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
from point_cloud import PointCloud

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Experiments] use python visualize_experiment.py config_file')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder)

    config_files = ['training_inference.json', 'test.json']
    for config_file in config_files:
        config = utils.read_json(config_folder + config_file)
        print('[Data] read ' + config_folder + config_file)

        n_observations = config['n_observations']
        statistics_file = common.filename(config, 'statistics_file', '.txt')

        with open(statistics_file, 'w') as f:

            targets = utils.read_hdf5(common.filename(config, 'output_file'))
            targets = np.repeat(targets, n_observations, axis=0)
            print('[Data] read ' + common.filename(config, 'output_file'))

            inputs = utils.read_hdf5(common.filename(config, 'input_file'))
            inputs = inputs.reshape((inputs.shape[0] * n_observations, 1, inputs.shape[2], inputs.shape[3], inputs.shape[4]))
            print('[Data] read ' + common.filename(config, 'input_file'))

            space = utils.read_hdf5(common.filename(config, 'space_file'))
            space = space.reshape((space.shape[0] * n_observations, 1, space.shape[2], space.shape[3], space.shape[4]))
            print('[Data] read ' + common.filename(config, 'space_file'))

            points = []
            point_dir = common.dirname(config, 'txt_gt_dir') + '/'

            for k in range(n_observations):
                k_point_dir = point_dir + '/%d/' % k
                for txt_file in os.listdir(k_point_dir):
                    point_cloud = PointCloud.from_txt(k_point_dir + txt_file)
                    points.append(point_cloud.points.shape[0])

            points = np.array(points)
            occupied = np.squeeze(np.sum(np.sum(np.sum(targets, axis=4), axis=3), axis=2))
            observed_points = np.squeeze(np.sum(np.sum(np.sum(inputs, axis=4), axis=3), axis=2))
            observed_space = np.squeeze(np.sum(np.sum(np.sum(space, axis=4), axis=3), axis=2))

            mask = np.zeros(inputs.shape)
            mask[inputs == 1] = 1
            mask[space == 1] = 1
            observed_total = float(np.sum(mask))

            mask = np.ones(targets.shape)
            mask[space == 0] = 0
            mask[targets == 0] = 0

            observed_invalid = float(np.sum(mask))

            f.write('[Data] set: ' + config_file + '\n')
            f.write('[Data] N: ' + str(targets.shape[0]) + '\n')
            f.write('[Data] voxels: ' + str(targets.shape[2] * targets.shape[3] * targets.shape[4]) + '\n')
            f.write('[Data] points: ' + str(np.mean(points)) + '\n')
            f.write('[Data] occupied: ' + str(np.sum(occupied) / targets.shape[0]) + '\n')
            f.write('[Data] observed points: ' + str(np.sum(observed_points) / targets.shape[0]) + '\n')
            f.write('[Data] observed space: ' + str(np.sum(observed_space) / targets.shape[0]) + '\n')
            f.write('[Data] observed total: ' + str(observed_total / targets.shape[0]) + '\n')
            f.write('[Data] observed invalid: ' + str(observed_invalid / targets.shape[0]) + '\n')

            print('[Data] set: ' + config_file)
            print('[Data] N: ' + str(targets.shape[0]))
            print('[Data] voxels: ' + str(targets.shape[2]*targets.shape[3]*targets.shape[4]))
            print('[Data] points: ' + str(np.mean(points)))
            print('[Data] occupied: ' + str(np.sum(occupied)/targets.shape[0]))
            print('[Data] observed points: ' + str(np.sum(observed_points)/targets.shape[0]))
            print('[Data] observed space: ' + str(np.sum(observed_space)/targets.shape[0]))
            print('[Data] observed total: ' + str(observed_total/targets.shape[0]))
            print('[Data] observed invalid: ' + str(observed_invalid/targets.shape[0]))

            for i in range(2, 5 + 1):

                # Check if multiple observations have been computed.
                if os.path.exists(common.filename(config, 'input_file', '.h5', i)) and \
                        os.path.exists(common.filename(config, 'space_file', '.h5', i)) and \
                        os.path.exists(common.dirname(config, 'txt_gt_dir', i)):

                    inputs = utils.read_hdf5(common.filename(config, 'input_file', '.h5', i))
                    inputs = inputs.reshape((inputs.shape[0] * n_observations, 1, inputs.shape[2], inputs.shape[3], inputs.shape[4]))
                    print('[Data] read ' + common.filename(config, 'input_file', '.h5', i))

                    space = utils.read_hdf5(common.filename(config, 'space_file', '.h5', i))
                    space = space.reshape((space.shape[0] * n_observations, 1, space.shape[2], space.shape[3], space.shape[4]))
                    print('[Data] read ' + common.filename(config, 'space_file', '.h5', i))

                    points = []
                    point_dir = common.dirname(config, 'txt_gt_dir', i) + '/'

                    for k in range(n_observations):
                        k_point_dir = point_dir + '/%d/' % k
                        for txt_file in os.listdir(k_point_dir):
                            point_cloud = PointCloud.from_txt(k_point_dir + txt_file)
                            points.append(point_cloud.points.shape[0])

                    points = np.array(points)
                    observed_points = np.squeeze(np.sum(np.sum(np.sum(inputs, axis=4), axis=3), axis=2))
                    observed_space = np.squeeze(np.sum(np.sum(np.sum(space, axis=4), axis=3), axis=2))

                    mask = np.zeros(inputs.shape)
                    mask[inputs == 1] = 1
                    mask[space == 1] = 1
                    observed_total = float(np.sum(mask))

                    mask = np.ones(targets.shape)
                    mask[space == 0] = 0
                    mask[targets == 0] = 0

                    observed_invalid = float(np.sum(mask))

                    f.write('[Data] ' + str(i) + ' points: ' + str(np.mean(points)) + '\n')
                    f.write('[Data] ' + str(i) + ' observed points: ' + str(np.sum(observed_points) / targets.shape[0]) + '\n')
                    f.write('[Data] ' + str(i) + ' observed space: ' + str(np.sum(observed_space) / targets.shape[0]) + '\n')
                    f.write('[Data] ' + str(i) + ' observed total: ' + str(observed_total / targets.shape[0]) + '\n')
                    f.write('[Data] ' + str(i) + ' observed invalid: ' + str(observed_invalid / targets.shape[0]) + '\n')

                    print('[Data] ' + str(i) + ' points: ' + str(np.mean(points)))
                    print('[Data] ' + str(i) + ' observed points: ' + str(np.sum(observed_points) / targets.shape[0]))
                    print('[Data] ' + str(i) + ' observed space: ' + str(np.sum(observed_space) / targets.shape[0]))
                    print('[Data] ' + str(i) + ' observed total: ' + str(observed_total / targets.shape[0]))
                    print('[Data] ' + str(i) + ' observed invalid: ' + str(observed_invalid / targets.shape[0]))
            print('[Data] wrote ' + statistics_file)