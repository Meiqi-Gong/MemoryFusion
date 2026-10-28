from .aot import AOT


def build_vos_model(name, cfg):
    if name != 'aot':
        raise ValueError('This release supports the MemoryFusion AOT model.')
    return AOT(cfg, encoder=cfg.MODEL_ENCODER)
