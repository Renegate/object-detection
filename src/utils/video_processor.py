import os

import cv2

from src.utils import Logger, Visualizer

logger = Logger.get_logger('VideoProcessor')


class VideoProcessor(object):
    def __init__(self, path, score_fn):

        self.score_fn = score_fn
        self.visualizer = Visualizer()

        if not os.path.exists(path):
            raise IOError('file %s does not exist'.format(path))
        self.capture = cv2.VideoCapture(path)

        while not self.capture.isOpened():
            cv2.waitKey(1000)
            logger.debug('Wait for header')

    def start(self, max_frame_num = 2 << 32, fps=1000):
        num_frames = min(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)), max_frame_num)
        logger.debug('process first {} frames'.format(num_frames))

        for i in xrange(num_frames):
            self._process_frame(fps)
        cv2.destroyAllWindows()

    def _process_frame(self, fps):
        flag, frame = self.capture.read()
        compacted = self.score_fn(frame)
        self.visualizer.draw(frame, compacted)
        cv2.imshow('video', frame)
        cv2.waitKey(max(1, 1000 / fps))
