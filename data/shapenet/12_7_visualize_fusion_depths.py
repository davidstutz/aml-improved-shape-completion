"""
Reconstruct.
"""

import os
import sys
from matplotlib import pyplot

sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
import utils
import common
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy import misc

def correct(filepath):
    from PIL import Image, ImageChops

    im = Image.open(filepath)
    pix = np.asarray(im)

    pix = pix[:, :, 0:3]  # Drop the alpha channel
    idx = np.where(pix - 255)[0:2]  # Drop the color when finding edges

    if idx[0].shape[0] > 0:
        box = map(min, idx)[::-1] + map(max, idx)[::-1]
        region = im.crop(box)
        region_pix = np.asarray(region)
        misc.imsave(filepath, region_pix)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Data] Usage python 10_reconstruct.py config_folder')
        exit(1)

    config_folder = sys.argv[1] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = ['test.json']
    for config_file in config_files:
        print('[Data] reading ' + config_folder + config_file)
        config = utils.read_json(config_folder + config_file)

        multiplier = config['multiplier']
        vis_directory = common.dirname(config, 'vis_dir')
        if not os.path.isdir(vis_directory):
            os.makedirs(vis_directory)

        depth_directory = config['depth_directory']
        depth_files = utils.read_ordered_directory(depth_directory)

        N = 30
        for i in range(N):
            n = i * (len(depth_files) * multiplier // N) // multiplier
            print('[Data] visualizing %d/%d' % ((n + 1), len(depth_files)))

            depths = utils.read_hdf5(depth_files[n])

            height = 3
            width = 6
            fig = plt.figure(figsize=(width * 1.6, height * 1.6))

            gs = matplotlib.gridspec.GridSpec(height, width)
            gs.update(wspace=0.025, hspace=0.025)

            for j in range(height*width):
                m = j*depths.shape[0]//(height*width)
                ax = plt.subplot(gs[j])
                ax.imshow(depths[m], cmap='coolwarm', interpolation='nearest', vmin=np.min(depths), vmax=np.max(depths))

            for i in range(height*width):
                ax = plt.subplot(gs[i])
                ax.axis('off')

            figure_file = vis_directory + '/%d_fd.png' % n
            plt.savefig(figure_file, bbox_inches='tight')
            print('[Experiments] wrote ' + figure_file)
            correct(figure_file)
