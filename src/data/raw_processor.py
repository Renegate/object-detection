from src.utils import Config, Logger
import cv2
import os
import ujson

logger = Logger.get_logger('Processor')

class RawProcessor(object):

    data_set_conf = Config.get('data_sets')

    @classmethod
    def download(cls, name):
        for conf in cls.data_set_conf:
            if conf.get('name') == name:

                data_set_dir = cls._get_raw_data_set_dir(name)
                url, compression_format = conf.get('url'), conf.get('compression_format')

                # skip download if data is present
                if os.path.exists(data_set_dir) and len(os.listdir(data_set_dir)) > 0:
                    logger.debug('Skip downloading, use cached files instead.')
                    return

                os.system('mkdir -p {}'.format(data_set_dir))
                os.system('wget {} -P {}'.format(url, data_set_dir))

                if (compression_format == 'zip'):
                    os.system('unzip {d}/*.zip -d {d} && rm -rf {d}/*.zip'.format(d=data_set_dir))
                return
        raise Exception('Data set {} not found in base.yaml'.format(name))

    @classmethod
    def load_raw_data(cls, names):
        '''
        load raw data into numpy ndarray
        :param names: names of data sets
        :return: map of {fname, ndarray}
        '''
        data_map = {}
        for name in names:
            for conf in cls.data_set_conf:
                if conf.get('name') == name:
                    data_set_dir = cls._get_raw_data_set_dir(name)
                    for file_name, full_file_name in cls._get_files_generator(
                            os.path.join(data_set_dir, conf.get('folder_name')),
                            conf.get('data_format')):
                        im = cv2.imread(os.path.join(data_set_dir, full_file_name))
                        data_map[file_name] = im
        return data_map

    @classmethod
    def load_raw_labels(cls, names):
        '''
        load raw labels into lists of [x, y, w, h, category]
        :param names: names of data sets
        :return: map of {fname, label_list}
        '''
        label_map = {}
        for name in names:
            for conf in cls.data_set_conf:
                if conf.get('name') == name:
                    data_set_dir = cls._get_raw_data_set_dir(name)
                    if conf.get('label_format') == 'idl':
                        # format {"60094.jpg": [[171.33312, 188.49996000000002, 243.8336, 240.66647999999998, 1]]}
                        for _, full_file_name in cls._get_files_generator(
                                os.path.join(data_set_dir, conf.get('folder_name')), 'idl'):
                            with open (full_file_name) as f:
                                for line in f:
                                    d = ujson.loads(line)
                                    for k in d:
                                        label_map[k] = d.get(k)
        return label_map

    @classmethod
    def _get_raw_data_set_dir(cls, name):
        return '{}/{}'.format(Config.get('data_raw_dir'), name)

    @classmethod
    def _get_files_generator(cls, directory, extension):
        """
        :param directory:
        :param extension:
        :return: a generator of tuples (file_name, full_file_name)
        """
        for dir_path, sub_dir_paths, files in os.walk(directory):
            for f in files:
                if f.endswith(extension):
                    yield f, os.path.join(dir_path, f)
