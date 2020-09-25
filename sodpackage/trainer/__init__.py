from . import SupervisedTrainer, DASupervisedTrainer, \
DiceSupervisedTrainer, MSDiceSupervisedTrainer, DADiceSupervisedTrainer, \
PreTrainingSupervisedTrainer

def get_trainer(config):
    return eval(config.TRAIN.TRAINER + '.' + config.TRAIN.TRAINER)
