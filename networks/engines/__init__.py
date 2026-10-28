from .aot_engine import AOTEngine, AOTInferEngine


def build_engine(name, phase='train', **kwargs):
    if name != 'aotengine' or phase not in ('train', 'eval'):
        raise ValueError(f'Unsupported engine/phase: {name}/{phase}')
    return (AOTEngine if phase == 'train' else AOTInferEngine)(**kwargs)
