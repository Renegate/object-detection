import os

from src.model.ssd.model_constants import ModelConstants
from src.utils import Config, Logger

logger = Logger.get_logger('BaseModel')


class BaseModel(object):
    def __init__(self, model_name):
        self.asset_dir = os.path.join(Config.get('models_dir'), model_name)
        os.system('mkdir -p {}'.format(self.asset_dir))
        self.asset_url_map = {}

        model_configs = Config.get('models')
        for conf in model_configs:
            if conf.get('name') == model_name:
                asset_urls = conf.get('asset_urls')
                for asset in asset_urls:
                    self.asset_url_map[asset['name']] = asset['url']

    def _download_asset(self, asset_name):

        logger.debug('Downloading asset: {}'.format(asset_name))
        full_asset_path = ModelConstants.FULL_ASSET_PATH

        if os.path.exists(full_asset_path):
            logger.debug('Skip downloading, use cached files instead.')
            return

        os.system('wget {} -O {}'.format(self.asset_url_map.get(asset_name), full_asset_path))

        if '.zip' in asset_name:
            os.system('unzip {} -d {}'.format(full_asset_path, self.asset_dir))

    def train(self, train_set, val_set):
        raise NotImplementedError()

    def test(self, test_set):
        raise NotImplementedError()

    def serve(self, instance):
        raise NotImplementedError()
