import os

import tensorflow as tf

from src.extern.nets import ssd_vgg_300
from src.extern.preprocessing import ssd_vgg_preprocessing
from src.model.base_model import BaseModel
from src.model.ssd.model_constants import ModelConstants
from src.utils import Logger

logger = Logger.get_logger('SSD')


class SSDModel(BaseModel):
    """ SSD Model """

    def __init__(self):
        BaseModel.__init__(self, ModelConstants.MODEL_NAME)

        self.session = None
        self.image_4d = None
        self.predictions = None
        self.localisations = None
        self.img_input = None  # tf placeholder
        self.bbox_img = None

    def train(self, train_set, val_set):
        logger.debug('Training ssd ...')
        logger.debug('Loading downloaded ssd model.')
        self._download_asset(ModelConstants.CHECKPOINT_PRETRAINED_FILE)
        self._load_checkpoint(ModelConstants.CHECKPOINT_PRETRAINED, False)
        # TODO: training
        self._save_checkpoint(ModelConstants.CHECKPOINT_TRAINED)

    def test(self, test_set):
        logger.debug('Testing ssd ...')
        logger.debug('Loading trained ssd model.')
        self._load_checkpoint(ModelConstants.CHECKPOINT_TRAINED, False)

        print len(test_set)
        for instance in test_set:
            # print instance
            rimg, rpredictions, rlocalisations, rbbox_img = self._score_instance(instance[1])
            # print rimg, rpredictions, rlocalisations, rbbox_img
            break

    def serve(self, instance):
        self._load_checkpoint(ModelConstants.CHECKPOINT_TRAINED, False)
        pass

    def _load_checkpoint(self, checkpoint, is_training=True):
        slim = tf.contrib.slim

        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.session = tf.Session(config=config)

        net_shape = (300, 300)
        data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.

        if is_training == True:
            # image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_train(
            #    self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.RESIZE_WARP_RESIZE)
            pass
        else:
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.RESIZE_WARP_RESIZE)

        self.image_4d = tf.expand_dims(image_pre, 0)

        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=is_training,
                                                                     reuse=False)

        # Restore SSD model.
        ckpt = os.path.join(ModelConstants.FULL_ASSET_PATH, checkpoint)

        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, ckpt)

    def _save_checkpoint(self, checkpoint):
        saver = tf.train.Saver()
        ckpt = os.path.join(ModelConstants.FULL_ASSET_PATH, checkpoint)
        saver.save(self.session, ckpt, write_meta_graph=False)

    def _summary(self):
        tf.summary.FileWriter(ModelConstants.FULL_ASSET_PATH, self.session.graph)

    def _score_instance(self, img):
        return self.session.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                         feed_dict={self.img_input: img})
