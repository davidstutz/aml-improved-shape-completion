import os
import sys
sys.path.insert(1, os.path.realpath('/BS/dstutz/work/shape-completion/code/lib/py/'))
import utils
import argparse
import mcubes


def get_parser():
    """
    Get parser.

    :return: parser
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser('Read LTSDF file and run marching cubes.')
    parser.add_argument('--input', type=str, help='Input HDF5 file.')
    parser.add_argument('--output', type=str, help='Output directory.')
    parser.add_argument('--n_observations', type=int, default=10, help='Number of observations per model, should be 10.')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    predictions = utils.read_hdf5(args.input)
    print('[Experiments] read ' + args.input)

    utils.makedir(args.output)
    print('[Experiments] created ' + args.output)

    for n in range(predictions.shape[0]):
        k = n%args.n_observations
        off_directory = args.output + '/%d/' % k
        utils.makedir(off_directory)

        off_file = off_directory + '/%d.off' % (n // args.n_observations)
        if not os.path.exists(off_file):
            vertices, triangles = mcubes.marching_cubes(-predictions[n][1].transpose(1, 0, 2), 0)
            mcubes.export_off(vertices, triangles, off_file)
            print('[Experiments] wrote ' + off_file)
