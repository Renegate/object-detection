from src.data import RawProcessor
from src.model import SSDModel
from src.utils import Config, Logger, Visualizer

logger = Logger.get_logger('TestHandler')


class TestHandler(object):
    data_sets = Config.get('test').get('data_sets', [])
    visualizer = Visualizer()

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
        if Config.get('model') == 'ssd':
            model = SSDModel()

        results = model.test(test_set)

        output_dir = Config.get('test').get('output_path')
        slide_show = Config.get('test').get('slide_show')
        json_lines = []

        for instance, result in zip(test_set, results):
            json_lines.append(cls._serialize(instance[0], result))
            if slide_show == True:
                cls.visualizer.draw(instance[1], result,
                                    show=True, wait_ms=2000, img_name=instance[0])

            with open(output_dir, 'w+') as f:
                f.writelines(json_lines)

    @classmethod
    def _serialize(self, key, result):
        """
        Neither json / ujson works. Implementing my own serializer.
        :return:
        """
        return '{{"{}": {}}}\n'.format(key, str(result))
