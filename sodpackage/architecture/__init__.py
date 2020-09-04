from . import HRNet, IHR, MultiIHR, ContrastiveMultiIHR
# print(locals())
def get_model(cfg):
    # get class
    Model = eval(cfg.MODEL.NAME + '.' + cfg.MODEL.NAME + '.' + cfg.MODEL.NAME)
    # get instance
    model = Model(cfg)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
