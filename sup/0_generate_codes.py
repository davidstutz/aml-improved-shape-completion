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

    parser = argparse.ArgumentParser()
    parser.add_argument('--code_size', type=int)
    parser.add_argument('--number', type=int)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    codes = np.random.randn(args.code_size, args.number)
    utils.write_hdf5('codes_' + str(args.code_size) + '_' + str(args.number) + '.h5', codes)
