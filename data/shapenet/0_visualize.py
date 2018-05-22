import argparse
import sys
import os
sys.path.insert(1, os.path.realpath(__file__ + '../lib/'))
from blender_utils import *
import json

def read_json(file):
    """
    Read a JSON file.

    :param file: path to file to read
    :type file: str
    :return: parsed JSON as dict
    :rtype: dict
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        return json.load(fp)

if __name__ == '__main__':
    try:
        argv = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        log('[Error] call as follows:', LogLevel.ERROR)
        log('[Error] $BLENDER --background --python 0_visualize.py -- 1>/dev/null config_folder', LogLevel.ERROR)
        exit()

    if len(argv) < 1:
        log('[Error] call as follows:', LogLevel.ERROR)
        log('[Error] $BLENDER --background --python 0_visualize.py -- 1>/dev/null config_folder', LogLevel.ERROR)
        exit()

    config_folder = argv[0] + '/'
    assert os.path.exists(config_folder), 'directory %s does not exist' % config_folder

    config_files = [config_file for config_file in os.listdir(config_folder)]
    config = read_json(config_folder + config_files[-1])

    raw_directory = config['raw_directory'] + '/'
    assert os.path.exists(raw_directory), 'directory %s does not exist' % raw_directory

    vis_directory = config['vis_directory'] + '/'
    if not os.path.isdir(vis_directory):
        os.makedirs(vis_directory)

    for filename in os.listdir(raw_directory):
        off_file = raw_directory + '/' + filename
        camera_target = initialize()

        off_material = make_material('BRC_Material_Mesh', (0.66, 0.45, 0.23), 0.8)
        load_off(off_file, off_material, (0, 0, 0), 1, 'xzy')

        rotations = [
            (0, 0, 30),
            (0, 0, 210),
        ]
        distance = 0.5

        for r in range(len(rotations)):
            png_file = vis_directory + '/%s_%d.png' % (os.path.splitext(filename)[0], r)
            render(camera_target, png_file, rotations[r], distance)
            log('[Data] wrote %s' % png_file)

        log('[Data] processed %s' % filename)
