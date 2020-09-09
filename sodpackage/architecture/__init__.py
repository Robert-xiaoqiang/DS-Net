from . import MRFNet, MSMRFNet, DAMRFNet
# print(locals())
def get_model(cfg):
    # get class by subpackage.module.classname
    Model = eval(cfg.MODEL.NAME + '.' + cfg.MODEL.NAME + '.' + cfg.MODEL.NAME)
    # get instance by instantiation
    model = Model(cfg)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
