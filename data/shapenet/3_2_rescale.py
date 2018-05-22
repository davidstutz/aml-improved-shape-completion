#!/usr/bin/env python
"""
Rescale.
"""

import os
import sys
import numpy as np
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import  utils
from mesh import Mesh

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 3_simplify.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    # Rescale all models in place (as this wasn't accounted for and there shouldn't be any more directories).
    config_files = [config_file for config_file in os.listdir(config_folder)]
    config = utils.read_json(config_folder + config_files[-1])

    padding = config['padding']
    simplified_directory = config['simplified_directory'] + '/'
    off_files = utils.read_ordered_directory(simplified_directory)

    for n in range(len(off_files)):
        mesh = Mesh.from_off(off_files[n])

        # Get extents of model.
        min, max = mesh.extents()
        total_min = np.min(np.array(min))
        total_max = np.max(np.array(max))

        # Set the center (although this should usually be the origin already).
        centers = (
            (min[0] + max[0]) / 2,
            (min[1] + max[1]) / 2,
            (min[2] + max[2]) / 2
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
            1 / (sizes[0] + 2 * padding * sizes[0]),
            1 / (sizes[1] + 2 * padding * sizes[1]),
            1 / (sizes[2] + 2 * padding * sizes[2])
        )

        mesh.translate(translation)
        mesh.scale(scales)

        print('[Data] (%s, %d) extents before %f - %f, %f - %f, %f - %f' % (off_files[n], n, min[0], max[0], min[1], max[1], min[2], max[2]))
        min, max = mesh.extents()
        print('[Data] (%s, %d) extents after %f - %f, %f - %f, %f - %f' % (off_files[n], n, min[0], max[0], min[1], max[1], min[2], max[2]))

        mesh.to_off(off_files[n])
