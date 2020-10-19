from . import MRFNet, MSMRFNet, DAMRFNet, \
RGB2DepthNet, D2DNet, D2DNetv2, D2DNetv3, D2DNetv4, D2DNetv5, D2DNetv6

# print(locals())
def get_model(cfg):
    # get class by subpackage.module.classname
    Model = eval(cfg.MODEL.NAME + '.' + cfg.MODEL.NAME + '.' + cfg.MODEL.NAME)
    # get instance by instantiation
    model = Model(cfg)
    model.init_weights(cfg.MODEL.PRETRAINED)
    if hasattr(cfg, 'PRETEXT') and hasattr(cfg.PRETEXT, 'PRETRAINED'):
        model.init_pretext(cfg.PRETEXT.PRETRAINED)

    return model
