"""
Scale.
"""

import numpy as np
import math

import os
import sys
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import  utils
from mesh import Mesh

def read_models(models_file):
    """
    Read the models file to get a list of filenames.

    :param models_file: models file path
    :type models_file: str
    :return: list of filenames
    :rtype: [str]
    """

    files = []
    with open(models_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line != '']
        files = [line.split(' ')[1] for line in lines]

    return files

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 1_scale.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder)]
    config = utils.read_json(config_folder + config_files[-1])

    raw_directory = config['raw_directory'] + '/'
    assert os.path.exists(raw_directory), 'directory %s does not exist' % raw_directory

    # The models file was crafted based on MatLab indexing; originally, the first file
    # in the list got the corresponding number, 1-based.
    # This means that the filter indices in the configuration file are 1-based.

    models_file = config['models_file']
    filenames = read_models(models_file)
    print('[Data] found %d files' % len(filenames))

    scaled_directory = config['scaled_directory'] + '/'
    utils.makedir(scaled_directory)

    padding = config['padding']
    y_rotation = config['y_rotation']
    blacklist_models = config['blacklist_models']
    whitelist_models = config['whitelist_models']

    j = 0
    for i in range(len(filenames)):

        if config['whitelist']:
            if (i + 1) not in whitelist_models:
                print('[Data] skipping (%d, %s)' % (i, filenames[i]))
                continue
        else:
            # Check if this model should be filtered; as above, note that the indices
            # in filter_models are 1-based.
            if (i + 1) in blacklist_models:
                print('[Data] skipping (%d, %s)' % (i, filenames[i]))
                continue

        in_file = raw_directory + filenames[i]
        if not os.path.exists(in_file):
            print('[Data] not found (%d, %s)' % (i, filenames[i]))
            continue

        mesh = Mesh.from_off(in_file)

        # Get extents of model.
        min, max = mesh.extents()
        total_min = np.min(np.array(min))
        total_max = np.max(np.array(max))

        # Set the center (although this should usually be the origin already).
        centers = (
            (min[0] + max[0])/2,
            (min[1] + max[1])/2,
            (min[2] + max[2])/2
        )
        # Scales all dimensions equally.
        sizes = (
            total_max - total_min,
            total_max - total_min,
            total_max - total_min
        )
        translation = (
            -centers[0],
            -centers[1],
            -centers[2]
        )
        scales = (
            1/(sizes[0] + 2 * padding*sizes[0]),
            1/(sizes[1] + 2 * padding*sizes[1]),
            1/(sizes[2] + 2 * padding*sizes[2])
        )

        mesh.translate(translation)
        mesh.scale(scales)

        print('[Data] (%s, %d) extents before %f - %f, %f - %f, %f - %f' % (filenames[i], i, min[0], max[0], min[1], max[1], min[2], max[2]))
        min, max = mesh.extents()
        print('[Data] (%s, %d) extents after %f - %f, %f - %f, %f - %f' % (filenames[i], i, min[0], max[0], min[1], max[1], min[2], max[2]))

        # Apply a fixed rotation.
        mesh.rotate((0, y_rotation/180.*math.pi, 0))

        out_file = scaled_directory + '/%d.off' % j
        mesh.to_off(out_file)
        j += 1