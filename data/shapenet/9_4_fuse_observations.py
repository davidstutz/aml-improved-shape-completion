"""
Post-process.
"""

import numpy as np
import os
import sys

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
from point_cloud import PointCloud

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 1_post_process.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    for config_file in os.listdir(config_folder):
        if config_file.find('training_prior') >= 0:
            continue;

        config = utils.read_json(config_folder + config_file)

        inputs = utils.read_hdf5(common.filename(config, 'input_file'))
        print('[Data] read ' + common.filename(config, 'input_file'))
        space = utils.read_hdf5(common.filename(config, 'space_file'))
        print('[Data] read ' + common.filename(config, 'space_file'))
        sdf_inputs = utils.read_hdf5(common.filename(config, 'input_sdf_file'))
        print('[Data] read ' + common.filename(config, 'input_sdf_file'))
        tsdf_inputs = utils.read_hdf5(common.filename(config, 'input_tsdf_file'))
        print('[Data] read ' + common.filename(config, 'input_tsdf_file'))
        ltsdf_inputs = utils.read_hdf5(common.filename(config, 'input_ltsdf_file'))
        print('[Data] read ' + common.filename(config, 'input_ltsdf_file'))

        N = inputs.shape[0]

        inputs = np.swapaxes(inputs, 0, 1)
        space = np.swapaxes(space, 0, 1)
        sdf_inputs = np.swapaxes(sdf_inputs, 0, 1)
        tsdf_inputs = np.swapaxes(tsdf_inputs, 0, 1)
        ltsdf_inputs = np.swapaxes(ltsdf_inputs, 0, 1)

        n_observations = config['n_observations']

        for i in range(2, 5 + 1):
            print('[Data] merging ' + str(i) + ' observations')

            i_inputs = np.zeros(inputs.shape)
            i_space = np.zeros(space.shape)
            i_sdf_inputs = np.zeros(sdf_inputs.shape)
            i_tsdf_inputs = np.zeros(tsdf_inputs.shape)
            i_ltsdf_inputs = np.zeros(ltsdf_inputs.shape)

            for k in range(n_observations):
                perm = np.random.permutation(n_observations)
                perm = perm[:i]
                print('[Data] perm ' + ', '.join(map(str, perm)))
                i_inputs[k] = np.sum(inputs[perm], axis=0)
                i_inputs[k] = np.clip(i_inputs[k], 0, 1)

                i_space[k] = np.sum(space[perm], axis=0)
                i_space[k] = np.clip(i_space[k], 0, 1)

                i_sdf_inputs[k] = np.min(sdf_inputs[perm], axis=0)
                i_tsdf_inputs[k] = np.min(tsdf_inputs[perm], axis=0)
                i_ltsdf_inputs[k] = np.min(ltsdf_inputs[perm], axis=0)

                # Also fuse the actual point clouds!
                txt_directories = utils.read_ordered_directory(common.dirname(config, 'txt_gt_dir'))
                txt_directory = common.dirname(config, 'txt_gt_dir', i) + '%d/' % k
                utils.makedir(txt_directory)

                for n in range(N):
                    point_cloud = PointCloud()
                    print('[Data] +')
                    for j in range(perm.shape[0]):
                        txt_file = txt_directories[perm[j]] + '/%d.txt' % n
                        print('[Data] | read ' + txt_file)
                        point_cloud_j = PointCloud.from_txt(txt_file)
                        point_cloud.points = np.concatenate((point_cloud.points, point_cloud_j.points), axis=0)

                    txt_file = txt_directory + '%d.txt' % n
                    point_cloud.to_txt(txt_file)
                    print('[Data] wrote ' + txt_file)

            i_inputs = np.swapaxes(i_inputs, 0, 1)
            i_space = np.swapaxes(i_space, 0, 1)
            i_sdf_inputs = np.swapaxes(i_sdf_inputs, 0, 1)
            i_tsdf_inputs = np.swapaxes(i_tsdf_inputs, 0, 1)
            i_ltsdf_inputs = np.swapaxes(i_ltsdf_inputs, 0, 1)

            print(i_inputs.shape)
            print(i_space.shape)
            print(i_sdf_inputs.shape)
            print(i_tsdf_inputs.shape)
            print(i_ltsdf_inputs.shape)

            utils.write_hdf5(common.filename(config, 'input_file', '.h5', i), i_inputs)
            print('[Data] wrote ' + common.filename(config, 'input_file', '.h5', i))
            utils.write_hdf5(common.filename(config, 'space_file', '.h5', i), i_space)
            print('[Data] wrote ' + common.filename(config, 'space_file', '.h5', i))
            utils.write_hdf5(common.filename(config, 'input_sdf_file', '.h5', i), i_sdf_inputs)
            print('[Data] wrote ' + common.filename(config, 'input_sdf_file', '.h5', i))
            utils.write_hdf5(common.filename(config, 'input_tsdf_file', '.h5', i), i_tsdf_inputs)
            print('[Data] wrote ' + common.filename(config, 'input_tsdf_file', '.h5', i))
            utils.write_hdf5(common.filename(config, 'input_ltsdf_file', '.h5', i), i_ltsdf_inputs)
            print('[Data] wrote ' + common.filename(config, 'input_ltsdf_file', '.h5', i))
