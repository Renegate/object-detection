import os
from src.model import SSDModel
from src.utils import Config, VideoProcessor

class ServeHandler(object):

    model = None

    @classmethod
    def handle(cls):

        cls.model = SSDModel()
        video_processor = VideoProcessor(os.path.join(Config.get('videos_dir'),
                                                      Config.get('serve').get('video')),
                                         cls.process)

        video_processor.start()

    @classmethod
    def process(cls, frame):
        # very slow
        return cls.model.serve((None, frame, None))

    @classmethod
    def process_precomputed(cls, frame):
        return
