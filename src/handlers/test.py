from src.data import RawProcessor
from src.model import SSDModel
from src.utils import Config, Logger

logger = Logger.get_logger('TestHandler')


class TestHandler(object):
    data_sets = Config.get('test').get('data_sets', [])

    @classmethod
    def handle(cls):
        cls._download()
        test_set = cls._process()
        cls._test(test_set)

    @classmethod
    def _download(cls):
        logger.debug('Fetching data sets: {}'.format(cls.data_sets))
        for name in cls.data_sets:
            RawProcessor.download(name)

    @classmethod
    def _process(cls):
        '''
        Load raw data as list of tuples.
        :return: None
        '''
        raw_data_map = RawProcessor.load_raw_data(cls.data_sets)
        return [(k, raw_data_map[k], None) for k in raw_data_map]

    @classmethod
    def _test(cls, test_set):

        model = None
        if Config.get('model') == 'yolov2':
            model = SSDModel()

        model.test(test_set)
