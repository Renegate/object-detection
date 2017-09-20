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
    # 21: 20, # traffic lights
}

RAW_TO_SSD_CLASS_MAPPING = {
    1: 7,   # vehicle
    2: 15,  # pedestrian
    3: 2,   # cyclist
    # 20: 21, # traffic lights
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
        self.visualizer = Visualizer()

    def train(self, train_set, val_set):
        logger.debug('Training ssd ...')
        logger.debug('Loading downloaded ssd model.')
        self._download_asset(ModelConstants.CHECKPOINT_PRETRAINED_FILE)
        self._load_checkpoint(ModelConstants.CHECKPOINT_PRETRAINED, True)
        # TODO: train

        # validation
        mAP = 0.0
        for instance in val_set:
            score = self._score_instance(instance[1])
            mAP += self._mean_average_precision(score, instance[2]) / float(len(val_set))

        logger.info('Pretrained checkpoint has mAP of {} on validation set'.format(mAP))

        self._save_checkpoint(ModelConstants.CHECKPOINT_TRAINED)

    def test(self, test_set, show=False):
        logger.debug('Testing ssd ...')
        logger.debug('Loading trained ssd model.')
        self._load_checkpoint(ModelConstants.CHECKPOINT_TRAINED, False)

        results = []

        for instance in test_set:
            result = self._score_instance(instance[1])
            if show == True:
                self.visualizer.draw(instance[1], result,
                                    show=True, wait_ms=2000, img_name=instance[0])
            results.append(result)

        return results

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

    def _save_summary(self):
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

    def _to_ssd_class(self, raw_class):
        return RAW_TO_SSD_CLASS_MAPPING.get(raw_class)

    def _to_raw_bbox(self, ssd_bbox, height, width):

        top_left_y = int(ssd_bbox[0] * height)
        top_left_x = int(ssd_bbox[1] * width)
        bot_right_y = int(ssd_bbox[2] * height)
        bot_right_x = int(ssd_bbox[3] * width)

        return top_left_x, top_left_y, bot_right_x, bot_right_y

    def _mean_average_precision(self, prediction, target):
        # format {cls: (prediction_labels, target_labels)}
        # where
        # prediction_labels: [(bbox1, confidence1), (bbox2, confidence2)],
        # target_labels: [(bbox1, confidence1), (bbox2, confidence2)]
        class_to_labels_map = dict.fromkeys(RAW_TO_SSD_CLASS_MAPPING.keys(), ([], []))
        mAP = 0.0

        for item in prediction:
            cls = item[4]
            if cls in class_to_labels_map:
                class_to_labels_map[cls][0].append((item[:4], item[5]))

        for item in target:
            cls = item[4]
            if cls in class_to_labels_map:
                class_to_labels_map[cls][1].append((item[:4], 1))

        for _, l in class_to_labels_map.iteritems():
            prediction_labels, target_labels = l
            prediction_labels = sorted(prediction_labels, cmp=lambda x1, x2: -1 if x2[1] > x1[1] else 1)

            AP = 0.0
            TP = [] # 1 for TP, 0 for FP
            for p in prediction_labels:
                # append FP by default
                TP.append(0)
                for t in target_labels:
                    if (self._iou(p[0], t[0]) > 0.5):
                        # change it to TP
                        TP[-1] = 1
                        break

                AP += float(sum(TP)) / float(len(TP)) / float(len(prediction_labels))
            mAP += AP / float(len(class_to_labels_map))
        return mAP

    def _iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

