from . import SupervisedTrainer, DASupervisedTrainer, \
DiceSupervisedTrainer, MSDiceSupervisedTrainer, DADiceSupervisedTrainer, \
PreTrainingSupervisedTrainer, MTSemiSupervisedTrainer

def get_trainer(config):
    return eval(config.TRAIN.TRAINER + '.' + config.TRAIN.TRAINER)
