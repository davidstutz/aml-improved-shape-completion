import time

class Timer:
    """
    Simple wrapper for time.clock().
    """

    def __init__(self):
        """
        Initialize and start timer.
        """

        self.start = time.clock()
        """ (float) Seconds. """

    def reset(self):
        """
        Reset timer.
        """

        self.start = time.clock()

    def elapsed(self):
        """
        Get elapsed time in seconds

        :return: elapsed time in seconds
        :rtype: float
        """

        return (time.clock() - self.start)

def filename(config, key, ext = '.h5', num = None):
    """
    Get the real file name by looking up the key in the config and suffixing.

    :param config: configuration read from JSON
    :type config: dict
    :param key: key to use in the config
    :type key: str
    :return: filepath
    :rtype: str
    """

    filepath = config[key] + '_' + str(config['multiplier']) + '_' \
           + str(config['image_height']) + 'x' + str(config['image_width']) + '_' \
           + str(config['height']) + 'x' + str(config['width']) + 'x' + str(config['depth'])

    if num is not None:
        filepath += '_' + str(num)

    return filepath + config['suffix'] + ext

def dirname(config, key, num = None):
    """
    Get the real directory name by looking up the key in the config and suffixing.

    :param config: configuration read from JSON
    :type config: dict
    :param key: key to use in the config
    :type key: str
    :return: dirpath
    :rtype: str
    """

    dirpath = config[key] + '_' + str(config['multiplier']) + '_' \
           + str(config['image_height']) + 'x' + str(config['image_width']) + '_' \
           + str(config['height']) + 'x' + str(config['width']) + 'x' + str(config['depth'])

    if num is not None:
        dirpath += '_' + str(num)

    return dirpath + config['suffix'] + '/' # !