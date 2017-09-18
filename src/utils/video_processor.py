import os

import cv2

from src.utils import Logger, Visualizer

logger = Logger.get_logger('VideoProcessor')


class VideoProcessor(object):
    def __init__(self, path, score_fn, annotated_path):

        self.score_fn = score_fn
        self.annotated_path = annotated_path
        self.visualizer = Visualizer()

        if not os.path.exists(path):
            raise IOError('file %s does not exist'.format(path))
        self.capture = cv2.VideoCapture(path)
        if os.path.exists(annotated_path):
            os.remove(annotated_path)
        self.writer = cv2.VideoWriter(annotated_path, cv2.VideoWriter_fourcc(*'XVID'),
                                      50.0, (640, 360))

        while not self.capture.isOpened():
            cv2.waitKey(1000)
            logger.debug('Wait for header')

    def start(self, max_frame_num = 2 << 32, fps=1000):
        num_frames = min(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)), max_frame_num)
        logger.debug('process first {} frames'.format(num_frames))

        for i in xrange(num_frames):
            self._process_frame(fps)
            if i % 100 == 0:
                logger.debug('processed {} frames.'.format(i))

        self.capture.release()
        self.writer.release()
        cv2.destroyAllWindows()

    def _process_frame(self, fps):
        flag, frame = self.capture.read()
        compacted = self.score_fn(frame)
        self.visualizer.draw(frame, compacted)
        self.writer.write(frame)
        cv2.imshow('video', frame)
        cv2.waitKey(max(1, 1000 / fps))
