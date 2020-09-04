import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
import argparse

from codpackage import architecture
from codpackage.datasampler.DataPreprocessor import DataPreprocessor
from codpackage.inference.Deducer import Deducer

from configure.default import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='/home/xqwang/projects/camouflaged/dev/configure/w48.yaml',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    parse_args()
    model = architecture.get_model(config)
    preprocessor = DataPreprocessor(config)
    test_dataloaders = preprocessor.get_test_dataloaders()

    deducer = Deducer(model, test_dataloaders, config)
    deducer.predict()

if __name__ == '__main__':
    main()