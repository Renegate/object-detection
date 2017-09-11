import os

import tensorflow as tf

from src.extern.nets import ssd_vgg_300, np_methods
from src.extern.preprocessing import ssd_vgg_preprocessing
from src.model.base_model import BaseModel
from src.model.ssd.model_constants import ModelConstants
from src.utils import Logger, Visualizer

SSD_TO_RAW_CLASS_MAPPING = {
    7: 1,   # vehicle
    15: 2,  # pedestrian
    2: 3,   # cyclist
    21: 20, # traffic lights
}

RAW_TO_SSD_CLASS_MAPPING = {
    1: 7,   # vehicle
    2: 15,  # pedestrian
    3: 2,   # cyclist
    20: 21, # traffic lights
}

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
        self.net_shape = (300, 300)
        self.ssd_anchors = None
        self.select_threshold=0.3
        self.nms_threshold=0.45

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

        return [self._score_instance(instance[1]) for instance in test_set]

    def serve(self, instance):
        if self.session is None:
            self._load_checkpoint(ModelConstants.CHECKPOINT_TRAINED, False)
        return self._score_instance(instance[1])

    def _load_checkpoint(self, checkpoint, is_training=True):
        slim = tf.contrib.slim

        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.session = tf.Session(config=config)

        data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.

        if is_training == True:
            # image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_train(
            #    self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.RESIZE_WARP_RESIZE)
            pass
        else:
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                self.img_input, None, None, self.net_shape, data_format, resize=ssd_vgg_preprocessing.RESIZE_WARP_RESIZE)

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
        self.ssd_anchors = ssd_net.anchors(self.net_shape)

    def _save_checkpoint(self, checkpoint):
        saver = tf.train.Saver()
        ckpt = os.path.join(ModelConstants.FULL_ASSET_PATH, checkpoint)
        saver.save(self.session, ckpt, write_meta_graph=False)

    def _summary(self):
        tf.summary.FileWriter(ModelConstants.FULL_ASSET_PATH, self.session.graph)

    def _score_instance(self, img):
        rimg, rpredictions, rlocalisations, rbbox_img = self.session.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                 feed_dict={self.img_input: img})

        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, self.ssd_anchors, select_threshold=self.select_threshold,
            img_shape=self.net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes,
                                                           nms_threshold=self.nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

        # return format:
        # [[top_left_x, top_left_y, bot_right_x, bot_right_y, class, confidence]]

        result = []
        for cls, score, bbox in zip(rclasses, rscores, rbboxes):
            raw_cls = self._to_raw_class(cls)
            if raw_cls is not None:
                top_left_x, top_left_y, \
                bot_right_x, bot_right_y = self._to_raw_bbox(bbox, img.shape[0], img.shape[1])
                result.append([top_left_x, top_left_y, bot_right_x, bot_right_y,
                               raw_cls, score])

        return result

    def _to_raw_class(self, ssd_class):
        return SSD_TO_RAW_CLASS_MAPPING.get(ssd_class)

    def _to_raw_bbox(self, ssd_bbox, height, width):

        top_left_y = int(ssd_bbox[0] * height)
        top_left_x = int(ssd_bbox[1] * width)
        bot_right_y = int(ssd_bbox[2] * height)
        bot_right_x = int(ssd_bbox[3] * width)

        return top_left_x, top_left_y, bot_right_x, bot_right_y
