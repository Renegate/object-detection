import tensorflow as tf
from src.utils import Config, Logger

class TFProcessor(object):

    @classmethod
    def read(cls, file_name):
        with tf.Session() as sess:
            pass

    @classmethod
    def write(cls, name, file, tuples):
        """
        Write data as tfrecords
        :param file_name: tfrecords file name
        :param tuples: list of (key, data, label)
        :return:
        """
        writer = tf.python_io.TFRecordWriter(file_name)
        '''
        for t in tuples:
            feature = {'key':
                       'data': _bytes_feature(tf.compat.as_bytes(t[1].tostring())),
                       'labels': }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        '''
        writer.close()

    @classmethod
    def _get_raw_data_set_dir(cls, name):
        return '{}/{}'.format(Config.get('data_tfrecords_dir'), name)

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
