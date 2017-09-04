from src.data import RawProcessor, TFProcessor
from src.utils import Config, Logger

logger = Logger.get_logger('TestHandler')

class TestHandler(object):

    data_sets = Config.get('train').get('data_sets', [])

    @classmethod
    def handle(cls):
        cls._download()
        cls._process()
        cls._test()

    @classmethod
    def _download(cls):
        logger.debug('Fetching data sets: {}'.format(cls.data_sets))
        for name in cls.data_sets:
            RawProcessor.download(name)

    @classmethod
    def _process(cls):
        '''
        Load raw data and store them as tfrecords format.
        Skip if tfrecords are present.
        :return: None
        '''
        raw_data = RawProcessor.load_raw_data(cls.data_sets)


    @classmethod
    def _test(cls):
        pass
