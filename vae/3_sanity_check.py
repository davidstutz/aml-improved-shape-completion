import os
import sys
sys.path.insert(1, os.path.realpath('../lib/py/'))
import utils
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import argparse


def get_parser():
    """
    Get parser.

    :return: parser
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser('Visualize predictions.')
    parser.add_argument('--predictions', type=str, help='Predictions HDF5 file.')
    parser.add_argument('--targets_occ', type=str, help='Ground truth occupancy grids HDF5 file.')
    parser.add_argument('--targets_sdf', type=str, help='Ground truth SDF HDF5 file.')
    parser.add_argument('--randoms', type=str, help='Random predictions HDF5 file.')
    parser.add_argument('--directory', type=str, help='Output directory.')
    parser.add_argument('--n_observations', type=int, default=10)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    targets_occ = utils.read_hdf5(args.targets_occ)
    # targets_occ = np.squeeze(targets_occ)
    print('[Experiments] read ' + args.targets_occ)

    targets_sdf = utils.read_hdf5(args.targets_sdf)
    # targets_sdf = np.squeeze(targets_sdf)
    print('[Experiments] read ' + args.targets_sdf)

    predictions = utils.read_hdf5(args.predictions)
    predictions = np.squeeze(predictions)
    print('[Experiments] read ' + args.predictions)

    randoms = utils.read_hdf5(args.randoms)
    randoms = np.squeeze(randoms)
    print('[Experiments] read ' + args.randoms)

    targets = np.concatenate((targets_occ, targets_sdf), axis=1)
    targets = targets[0:predictions.shape[0]]
    error = np.abs(predictions - targets)
    print('[Experiments] computed error')

    min_value = 0  # min(np.min(predictions[0:len(indices)//2]), np.min(randoms[0:len(indices)]))
    max_value = 1  # max(np.max(predictions[0:len(indices)//2]), np.max(randoms[0:len(indices)]))

    slices = []
    for i in range(targets.shape[2] // 2 - 2):
        slices.append(i)

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    for k in range(5):
        plt.clf()
        fig = plt.figure(figsize=(len(slices) * 1.6, 6 * 1.6))

        gs = matplotlib.gridspec.GridSpec(6, len(slices))
        gs.update(wspace=0.025, hspace=0.025)

        for i in slices:
            n = 6 * k + 0
            ax = plt.subplot(gs[0 * len(slices) + i])
            ax.imshow(randoms[n][0][4 + 2 * i], cmap='coolwarm', interpolation='nearest', vmin=min_value,
                      vmax=max_value)
        for i in slices:
            n = 6 * k + 0
            ax = plt.subplot(gs[1 * len(slices) + i])
            im = ax.imshow(randoms[n][1][4 + 2 * i], cmap='coolwarm', interpolation='nearest',
                           vmin=np.min(randoms[n][1]), vmax=np.max(randoms[n][1]))
        for i in slices:
            n = 6 * k + 1
            ax = plt.subplot(gs[2 * len(slices) + i])
            im = ax.imshow(randoms[n][0][4 + 2 * i], cmap='coolwarm', interpolation='nearest', vmin=min_value,
                           vmax=max_value)
        for i in slices:
            n = 6 * k + 1
            ax = plt.subplot(gs[3 * len(slices) + i])
            im = ax.imshow(randoms[n][1][4 + 2 * i], cmap='coolwarm', interpolation='nearest',
                           vmin=np.min(randoms[n][1]), vmax=np.max(randoms[n][1]))
        for i in slices:
            n = 6 * k + 2
            ax = plt.subplot(gs[4 * len(slices) + i])
            im = ax.imshow(randoms[n][0][4 + 2 * i], cmap='coolwarm', interpolation='nearest', vmin=min_value,
                           vmax=max_value)
        for i in slices:
            n = 6 * k + 2
            ax = plt.subplot(gs[5 * len(slices) + i])
            im = ax.imshow(randoms[n][1][4 + 2 * i], cmap='coolwarm', interpolation='nearest',
                           vmin=np.min(randoms[n][1]), vmax=np.max(randoms[n][1]))

        # https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-x-or-y-axis-in-matplotlib
        for i in range(6 * len(slices)):
            ax = plt.subplot(gs[i])
            ax.axis('off')

        figure_file = args.directory + '/random_%d.png' % k
        plt.savefig(figure_file, bbox_inches='tight')
        print('[Experiments] wrote ' + figure_file)

    # !
    predictions = np.repeat(predictions, args.n_observations, axis=0)

    N = 30
    for i in range(N):
        n = i * predictions.shape[0] // N
        fig = plt.figure(figsize=(len(slices) * 1.6, 6 * 1.6))

        gs = matplotlib.gridspec.GridSpec(6, len(slices))
        gs.update(wspace=0.025, hspace=0.025)

        for i in slices:
            ax = plt.subplot(gs[0 * len(slices) + i])
            ax.imshow(targets[n][0][4 + 2 * i], cmap='coolwarm', interpolation='nearest', vmin=min_value,
                      vmax=max_value)
        for i in slices:
            ax = plt.subplot(gs[1 * len(slices) + i])
            ax.imshow(targets[n][1][4 + 2 * i], cmap='coolwarm', interpolation='nearest', vmin=np.min(targets[n][1]),
                      vmax=np.max(targets[n][1]))
        for i in slices:
            ax = plt.subplot(gs[2 * len(slices) + i])
            ax.imshow(predictions[n][0][4 + 2 * i], cmap='coolwarm', interpolation='nearest', vmin=min_value,
                      vmax=max_value)
        for i in slices:
            ax = plt.subplot(gs[3 * len(slices) + i])
            ax.imshow(predictions[n][1][4 + 2 * i], cmap='coolwarm', interpolation='nearest',
                      vmin=np.min(predictions[n][1]), vmax=np.max(predictions[n][1]))
        for i in slices:
            ax = plt.subplot(gs[4 * len(slices) + i])
            ax.imshow(error[n][0][4 + 2 * i], cmap='coolwarm', interpolation='nearest', vmin=min_value, vmax=max_value)
        for i in slices:
            ax = plt.subplot(gs[5 * len(slices) + i])
            ax.imshow(error[n][1][4 + 2 * i], cmap='coolwarm', interpolation='nearest', vmin=np.min(error[n][1]),
                      vmax=np.max(error[n][1]))

        for i in range(6 * len(slices)):
            ax = plt.subplot(gs[i])
            ax.axis('off')

        figure_file = args.directory + '/results_%d.png' % n
        plt.savefig(figure_file, bbox_inches='tight')
        print('[Experiments] wrote ' + figure_file)
