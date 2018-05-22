import os
import sys
sys.path.insert(1, os.path.realpath('../lib/py/'))
import utils
import argparse
import numpy as np


def get_parser():
    """
    Get parser.

    :return: parser
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser('Evaluate predicted occupancy grids and LTSDFs.')
    parser.add_argument('--predictions', type=str, help='Predictions HDF5 file.')
    parser.add_argument('--targets_occ', type=str, help='Ground truth occupancy grids as HDF5 file.')
    parser.add_argument('--targets_sdf', type=str, help='Ground truth LTSDF as HDF5 file.')
    parser.add_argument('--results_file', type=str, help='Results txt file.')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    assert os.path.exists(args.predictions)
    predictions = utils.read_hdf5(args.predictions)

    assert os.path.exists(args.targets_occ)
    targets_occ = utils.read_hdf5(args.targets_occ)
    targets_occ = np.squeeze(targets_occ)

    assert os.path.exists(args.targets_sdf)
    targets_sdf = utils.read_hdf5(args.targets_sdf)
    targets_sdf = np.squeeze(targets_sdf)

    predictions_occ = predictions[:, 0]
    predictions_sdf = predictions[:, 1]

    predictions_occ_thresh = predictions_occ.copy()
    predictions_occ_thresh[predictions_occ_thresh > 0.5] = 1
    predictions_occ_thresh[predictions_occ_thresh < 0.5] = 0

    predictions_sdf_thresh = predictions_occ.copy()
    predictions_sdf_thresh[predictions_sdf_thresh > 0] = 0
    predictions_sdf_thresh[predictions_sdf_thresh < 0] = 1

    occ_error = np.sum(np.abs(predictions_occ - targets_occ)) / (predictions.shape[0]*predictions.shape[2]*predictions.shape[3]*predictions.shape[4])
    occ_error_thresh = np.sum(np.abs(predictions_occ_thresh - targets_occ)) / (predictions.shape[0] * predictions.shape[2] * predictions.shape[3] * predictions.shape[4])
    sdf_error = np.sum(np.abs(predictions_sdf - targets_sdf)) / (predictions.shape[0] * predictions.shape[2] * predictions.shape[3] * predictions.shape[4])
    sdf_error_thresh = np.sum(np.abs(predictions_sdf_thresh - targets_occ)) / (predictions.shape[0] * predictions.shape[2] * predictions.shape[3] * predictions.shape[4])

    occ_intersection = np.zeros((predictions_occ.shape))
    occ_intersection[predictions_occ_thresh + targets_occ > 1] = 1

    occ_union = np.zeros((predictions_occ.shape))
    occ_union[(predictions_occ_thresh + targets_occ) > 0] = 1
    occ_iou_thresh = np.sum(occ_intersection) / np.sum(occ_union)

    sdf_intersection = np.zeros((predictions_sdf.shape))
    sdf_intersection[(predictions_sdf_thresh + targets_sdf) > 1] = 1

    sdf_union = np.zeros((predictions_sdf.shape))
    sdf_union[predictions_sdf_thresh + targets_sdf > 0] = 1
    sdf_iou_thresh = np.sum(sdf_intersection) / np.sum(sdf_union)

    with open(args.results_file, 'w') as f:
        f.write('[Experiments] occ error: ' + str(occ_error) + '\n')
        f.write('[Experiments] occ error+thresh: ' + str(occ_error_thresh) + '\n')
        f.write('[Experiments] sdf error: ' + str(sdf_error) + '\n')
        f.write('[Experiments] sdf error+thresh: ' + str(sdf_error_thresh) + '\n')
        f.write('[Experiments] occ IoU: ' + str(occ_iou_thresh) + '\n')
        f.write('[Experiments] sdf IoU: ' + str(sdf_iou_thresh) + '\n')

        print('[Experiments] occ error: ' + str(occ_error))
        print('[Experiments] occ error+thresh: ' + str(occ_error_thresh))
        print('[Experiments] sdf error: ' + str(sdf_error))
        print('[Experiments] sdf error+thresh: ' + str(sdf_error_thresh))
        print('[Experiments] occ IoU: ' + str(occ_iou_thresh))
        print('[Experiments] sdf IoU: ' + str(sdf_iou_thresh))
