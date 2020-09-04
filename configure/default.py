import os

from yacs.config import CfgNode as CN

_C = CN()

_C.NAME = ''
_C.DEVICE = ''
_C.SEED = 32767
_C.SUMMARY_DIR = ''

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'HRNet'
_C.MODEL.NONLOCAL_TYPE = 'none'
_C.MODEL.PRETRAINED = ''
_C.MODEL.CONTRASTIVE = CN(new_allowed=True)
_C.MODEL.EXTRA = CN(new_allowed=True)

# training
_C.TRAIN = CN()
_C.TRAIN.TRAINER = ''
_C.TRAIN.DATASET_ROOT = ''
_C.TRAIN.TRAIN_SIZE = [352, 352]  # width * height
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.WORKERS = 4
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_EPOCHS = 288
_C.TRAIN.RESUME = True
_C.TRAIN.LOSS_FREQ = 10
_C.TRAIN.TB_FREQ = 10

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIM = 'sgd_all'
_C.TRAIN.LR = 0.01
_C.TRAIN.LD = 0.9
_C.TRAIN.WD = 5.0e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV = False
_C.TRAIN.REDUCTION = 'mean'

# validating
_C.VAL = CN()
_C.VAL.DATASET_ROOT = ''

# testing
_C.TEST = CN()
_C.TEST.DATASET_ROOTS = [ ]
_C.TEST.BATCH_SIZE = 12
_C.TEST.WORKERS = 8

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False

from .models import MODEL_EXTRAS
_C.EXTRA = MODEL_EXTRAS['seg_hrnet']
config = _C

def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

if __name__ == '__main__':
    import sys
    import os
    file_name = os.path.join(os.path.dirname(__file__), sys.argv[1])
    with open(filename, 'w') as f:
        print(_C, file=f)
