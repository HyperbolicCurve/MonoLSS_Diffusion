from lib.models.MonoLSS import MonoLSS


def build_model(cfg, mean_size):
    if cfg['type'] == 'MonoLSS':
        return MonoLSS(config=cfg, backbone=cfg['backbone'], mean_size=mean_size)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
