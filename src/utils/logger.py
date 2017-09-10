from config import Config
import logging

class Logger(object):

    """
    Wrapper class on logging
    """

    @classmethod
    def get_logger(cls, name):

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')

        _logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

        debug_enabled = Config.get('logging', 'info') == 'debug'

        if debug_enabled == True:
            _logger.setLevel(logging.DEBUG)
        else:
            _logger.setLevel(logging.INFO)
        return _logger
