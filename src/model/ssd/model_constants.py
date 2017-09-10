import os

from src.utils import Config


class ModelConstants(object):
    MODEL_NAME = 'ssd'
    CHECKPOINT_PRETRAINED_FILE = 'ssd_300_vgg.ckpt.zip'
    CHECKPOINT_PRETRAINED = 'ssd_300_vgg.ckpt'
    CHECKPOINT_TRAINED = 'ssd_trained.ckpt'

    FULL_ASSET_PATH = os.path.join(Config.get('models_dir'), MODEL_NAME)
