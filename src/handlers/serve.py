import os

from src.model import SSDModel
from src.utils import Config, Logger, VideoProcessor

logger = Logger.get_logger('ServeHandler')


class ServeHandler(object):
    model = None

    @classmethod
    def handle(cls):

        if Config.get('model') == 'ssd':
            cls.model = SSDModel()

        logger.debug('Start serving ...')
        video_path = os.path.join(Config.get('videos_dir'),
                                  Config.get('serve').get('video'))

        if os.path.exists(video_path):
            logger.debug('video already exists, skip downloading')
        else:
            url = None
            confs = Config.get('videos')
            for conf in confs:
                if conf.get('name') == Config.get('serve').get('video'):
                    url = conf.get('url')
                    break

            os.system('wget {} -O {}'.format(url, video_path))

        video_processor = VideoProcessor(video_path, cls.process)

        video_processor.start()

    @classmethod
    def process(cls, frame):
        # very slow
        return cls.model.serve((None, frame, None))

    @classmethod
    def process_precomputed(cls, frame):
        return
