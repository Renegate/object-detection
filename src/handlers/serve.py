import os
import ujson
from src.model import SSDModel
from src.utils import Config, Logger, VideoProcessor

logger = Logger.get_logger('ServeHandler')


class ServeHandler(object):
    model = None
    scores = []
    frame_cnt = 0
    use_precomputed = False

    @classmethod
    def handle(cls):

        if Config.get('model') == 'ssd':
            cls.model = SSDModel()

        logger.debug('Start serving ...')
        full_video_path = os.path.join(Config.get('videos_dir'),
                                  Config.get('serve').get('video'))

        url = None
        precomputed_labels = None
        confs = Config.get('videos')
        for conf in confs:
            if conf.get('name') == Config.get('serve').get('video'):
                url = conf.get('url')
                precomputed_labels = conf.get('precomputed_labels')
                break

        # download video if necessary
        if os.path.exists(full_video_path):
            logger.debug('video already exists, skip downloading')
        else:
            os.system('curl {} --create-dirs -o {}'.format(url, full_video_path))

        # load precomputed labels if possible
        precomputed_labels_path = os.path.join(Config.get('videos_dir'), precomputed_labels)
        if os.path.exists(precomputed_labels_path):
            cls.use_precomputed = True
            with open(precomputed_labels_path, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    cls.scores.append(ujson.loads(l))
            logger.debug('precomputed labels file exists, skip real time prediction')

        score_fn = cls.process_precomputed if cls.use_precomputed == True else cls.process
        fps = 50 if cls.use_precomputed == True else 1000

        video_processor = VideoProcessor(full_video_path, score_fn)
        video_processor.start(max_frame_num=Config.get('serve').get('max_frame_num'), fps=fps)

        if cls.use_precomputed == False and len(cls.scores) > 0:
            with open (precomputed_labels_path, 'w+') as f:
                for score in cls.scores:
                    f.write(str(score) + '\n')

    @classmethod
    def process(cls, frame):
        # very slow
        compacted = cls.model.serve((None, frame, None))
        cls.scores.append(compacted)
        return compacted

    @classmethod
    def process_precomputed(cls, frame):
        score = cls.scores[cls.frame_cnt]
        cls.frame_cnt += 1
        return score
