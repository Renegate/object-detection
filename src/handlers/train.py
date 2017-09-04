from random import seed, shuffle
from src.data import RawProcessor, TFProcessor
from src.utils import Config, Logger

logger = Logger.get_logger('TrainHandler')

class TrainHandler(object):

    data_sets = Config.get('train').get('data_sets', [])
    holdout_percentage = Config.get('holdout_percentage', 0.1)

    @classmethod
    def handle(cls):
        cls._download()
        cls._process()
        cls._train()

    @classmethod
    def _download(cls):
        logger.debug('Fetching data sets: {}'.format(cls.data_sets))
        for name in cls.data_sets:
            RawProcessor.download(name)

    @classmethod
    def _process(cls):
        '''
        Load raw data and labels, split them into training sets and validation sets.
        And store them as tfrecords format. Skip if tfrecords are present.
        :return: None
        '''
        raw_data_map = RawProcessor.load_raw_data(cls.data_sets)
        raw_label_map = RawProcessor.load_raw_labels(cls.data_sets)

        seed(0)
        shuffled_keys = [k for k in raw_data_map]
        shuffle(shuffled_keys)

        split_index = int(round(len(shuffled_keys) * (1 - cls.holdout_percentage)))
        train_keys = shuffled_keys[: split_index]
        validation_keys = shuffled_keys[split_index :]

        train_set = [(k, raw_data_map[k], raw_label_map[k]) for k in train_keys]
        validation_set = [(k, raw_data_map[k], raw_label_map[k]) for k in validation_keys]

        TFProcessor.write('', train_set)
        TFProcessor.write('', validation_set)

    @classmethod
    def _train(cls):
        pass
