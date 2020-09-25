import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
import argparse

from sodpackage import architecture
from sodpackage.datasampler.DataPreprocessor import DataPreprocessor
from sodpackage import trainer

from configure.default import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='/home/xqwang/projects/saliency/dev/configure/w18-baseline-0.yaml',
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
    # get instance
    model = architecture.get_model(config)

    preprocessor = DataPreprocessor(config)
    train_dataloader = preprocessor.get_train_dataloader()
    val_dataloader = preprocessor.get_val_dataloader()
    test_dataloaders = preprocessor.get_test_dataloaders()
    
    # get class
    Trainer = trainer.get_trainer(config)
    # instantiate
    t = Trainer(model, train_dataloader, val_dataloader, test_dataloaders, config)
    t.train()

if __name__ == '__main__':
    main()
