from src.handlers import TrainHandler, TestHandler, ServeHandler
from src.utils import ArgParser, Constants

def main(context):
    if context.get(Constants.MODE) == Constants.MODE_TRAIN:
        TrainHandler.handle()
    elif context.get(Constants.MODE) == Constants.MODE_TEST:
        TestHandler.handle()
    elif context.get(Constants.MODE) == Constants.MODE_SERVE:
        ServeHandler.handle()

if __name__ == '__main__':
    context = ArgParser.parse()
    main(context)
