from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='MemoryFusion', stage='memoryfusion', model='r50_aotl'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = stage
        self.DATASETS = ['M3SVD']
        self.TRAIN_TOTAL_STEPS = 1000000
        self.DATA_SEQ_LEN = 30
        self.TRAIN_LONG_TERM_MEM_GAP = 3
        self.MODEL_LINEAR_Q = False
        self.MODEL_IGNORE_TOKEN = True
        self.TRAIN_AUTO_RESUME = False
        self.PRETRAIN = False
        self.PRETRAIN_FULL = True
        self.PRETRAIN_MODEL = ''
        self.init_dir()
