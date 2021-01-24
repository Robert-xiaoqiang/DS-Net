from . import SupervisedTrainer, DASupervisedTrainer, \
DiceSupervisedTrainer, MSDiceSupervisedTrainer, DADiceSupervisedTrainer, \
PreTrainingSupervisedTrainer, MTSemiSupervisedTrainer, DADisentangleSupervisedTrainer, \
MTDisentangleSemiSupervisedTrainer, GCMTDisentangleSemiSupervisedTrainer

def get_trainer(config):
    return eval(config.TRAIN.TRAINER + '.' + config.TRAIN.TRAINER)
