from random import seed, shuffle
from src.data import RawProcessor
from src.model import SSDModel
from src.utils import Config, Logger

logger = Logger.get_logger('TrainHandler')

class TrainHandler(object):

    data_sets = Config.get('train').get('data_sets', [])
    holdout_percentage = Config.get('holdout_percentage', 0.1)

    @classmethod
    def handle(cls):
        cls._download()
        train_set, val_set = cls._process()
        cls._train(train_set, val_set)

    @classmethod
    def _download(cls):
        logger.debug('Fetching data sets: {}'.format(cls.data_sets))
        for name in cls.data_sets:
            RawProcessor.download(name)

    @classmethod
    def _process(cls):
        '''
        Load raw data and labels, split them into training sets and validation sets.
        :return: None
        '''
        raw_data_map = RawProcessor.load_raw_data(cls.data_sets)
        raw_label_map = RawProcessor.load_raw_labels(cls.data_sets)

        seed(0)
        shuffled_keys = [k for k in raw_data_map]
        shuffle(shuffled_keys)

        split_index = int(round(len(shuffled_keys) * (1 - cls.holdout_percentage)))
        train_keys = shuffled_keys[: split_index]
        val_keys = shuffled_keys[split_index :]

        train_set = [(k, raw_data_map[k], raw_label_map[k]) for k in train_keys]
        val_set = [(k, raw_data_map[k], raw_label_map[k]) for k in val_keys]

        return train_set, val_set

    @classmethod
    def _train(cls, train_set, val_set):

        model = None
        if Config.get('model') == 'ssd':
            model = SSDModel()

        model.train(train_set, val_set)
