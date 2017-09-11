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

    def start(self):
        while True:
            self._process_frame()

    def _process_frame(self):
        flag, frame = self.capture.read()
        compacted = self.score_fn(frame)
        self.visualizer.draw(frame, compacted)
        cv2.imshow('video', frame)
        cv2.waitKey(1)
