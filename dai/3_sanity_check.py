import os
import sys
sys.path.insert(1, os.path.realpath('/BS/dstutz/work/shape-completion/code/lib/py/'))
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

    parser = argparse.ArgumentParser('Sanity check predictions.')
    parser.add_argument('--inputs_occ', type=str, help='Input, i.e. observations, as occupancy grids HDF5 file.')
    parser.add_argument('--inputs_sdf', type=str, help='Input/observations as LTSDFs HDF5 file.')
    parser.add_argument('--predictions', type=str, help='Predictions HDF5 file.')
    parser.add_argument('--targets_occ', type=str, help='Ground truth occupancy grids HDF5 file.')
    parser.add_argument('--targets_sdf', type=str, help='Ground truth LTSDF HDF5 file.')
    parser.add_argument('--directory', type=str, help='Output directory.')
    parser.add_argument('--n_observations', type=int, default=10, help='Number of observations per model, should be 10.')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    inputs_occ = utils.read_hdf5(args.inputs_occ)
    print('[Experiments] read ' + args.inputs_occ)

    inputs_sdf = utils.read_hdf5(args.inputs_sdf)
    print('[Experiments] read ' + args.inputs_sdf)

    targets_occ = utils.read_hdf5(args.targets_occ)
    print('[Experiments] read ' + args.targets_occ)

    targets_sdf = utils.read_hdf5(args.targets_sdf)
    print('[Experiments] read ' + args.targets_sdf)

    predictions = utils.read_hdf5(args.predictions)
    predictions = np.squeeze(predictions)
    print('[Experiments] read ' + args.predictions)

    if len(predictions.shape) > 5:
        predictions = predictions[:, 0, :, :, :, :]

    # Note that we have multiple observations per target,
    # so targets need to be repeated.
    targets_occ = np.repeat(targets_occ, args.n_observations, axis=0)
    targets_sdf = np.repeat(targets_sdf, args.n_observations, axis=0)

    targets_occ = targets_occ[:predictions.shape[0]]
    targets_sdf = targets_sdf[:predictions.shape[0]]

    #assert targets_occ.shape[0] == predictions.shape[0]
    #assert targets_sdf.shape[0] == predictions.shape[0]

    inputs_occ = inputs_occ.reshape((inputs_occ.shape[0]*inputs_occ.shape[1], 1, inputs_occ.shape[2], inputs_occ.shape[3], inputs_occ.shape[4]))
    inputs_sdf = inputs_sdf.reshape((inputs_sdf.shape[0]*inputs_sdf.shape[1], 1, inputs_sdf.shape[2], inputs_sdf.shape[3], inputs_sdf.shape[4]))

    inputs_occ = inputs_occ[:predictions.shape[0]]
    inputs_sdf = inputs_sdf[:predictions.shape[0]]

    #assert inputs_occ.shape[0] == predictions.shape[0], 'inputs_occ %d and predictions %d' % (inputs_occ.shape[0], predictions.shape[0])
    #assert inputs_sdf.shape[0] == predictions.shape[0]

    targets = np.concatenate((targets_occ, targets_sdf), axis=1)
    inputs = np.concatenate((inputs_occ, inputs_sdf), axis=1)

    error = np.abs(predictions - targets)
    print('[Experiments] computed error')

    min_value = 0#min(np.min(predictions[0:len(indices)//2]), np.min(randoms[0:len(indices)]))
    max_value = 1#max(np.max(predictions[0:len(indices)//2]), np.max(randoms[0:len(indices)]))

    slices = []
    for i in range(targets.shape[2]//2 - 2):
        slices.append(i)

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    N = 30
    for i in range(N):
        n = i * predictions.shape[0] // N
        fig = plt.figure(figsize=(len(slices)*1.6, 8*1.6))

        gs = matplotlib.gridspec.GridSpec(8, len(slices))
        gs.update(wspace=0.025, hspace=0.025)

        for i in slices:
            ax = plt.subplot(gs[0 * len(slices) + i])
            ax.imshow(inputs[n][0][4 + 2*i], cmap='coolwarm', interpolation='nearest', vmin=min_value, vmax=max_value)
        for i in slices:
            ax = plt.subplot(gs[1 * len(slices) + i])
            ax.imshow(inputs[n][1][4 + 2*i], cmap='coolwarm', interpolation='nearest', vmin=np.min(targets[n][1]), vmax=np.max(targets[n][1]))
        for i in slices:
            ax = plt.subplot(gs[2 * len(slices) + i])
            ax.imshow(predictions[n][0][4 + 2*i], cmap='coolwarm', interpolation='nearest', vmin=min_value, vmax=max_value)
        for i in slices:
            ax = plt.subplot(gs[3 * len(slices) + i])
            ax.imshow(predictions[n][1][4 + 2*i], cmap='coolwarm', interpolation='nearest', vmin=np.min(predictions[n][1]), vmax=np.max(predictions[n][1]))
        for i in slices:
            ax = plt.subplot(gs[4 * len(slices) + i])
            ax.imshow(targets[n][0][4 + 2*i], cmap='coolwarm', interpolation='nearest', vmin=min_value, vmax=max_value)
        for i in slices:
            ax = plt.subplot(gs[5 * len(slices) + i])
            ax.imshow(targets[n][1][4 + 2*i], cmap='coolwarm', interpolation='nearest', vmin=np.min(targets[n][1]), vmax=np.max(targets[n][1]))
        for i in slices:
            ax = plt.subplot(gs[6 * len(slices) + i])
            ax.imshow(error[n][0][4 + 2*i], cmap='coolwarm', interpolation='nearest', vmin=min_value, vmax=max_value)
        for i in slices:
            ax = plt.subplot(gs[7 * len(slices) + i])
            ax.imshow(error[n][1][4 + 2*i], cmap='coolwarm', interpolation='nearest', vmin=np.min(error[n][1]), vmax=np.max(error[n][1]))

        for i in range(8*len(slices)):
            ax = plt.subplot(gs[i])
            ax.axis('off')

        figure_file = args.directory + '/results_%d.png' % n
        plt.savefig(figure_file, bbox_inches='tight')
        print('[Experiments] wrote ' + figure_file)
