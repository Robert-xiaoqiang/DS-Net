from . import Deducer, PreTrainingDeducer

def get_deducer(cfg):
    # get class by module.classname
    return eval(cfg.TEST.DEDUCER + '.' + cfg.TEST.DEDUCER)
