class DefaultModelConfig:

    def __init__(self):
        self.MODEL_NAME = 'AOTDefault'
        self.MODEL_VOS = 'aot'
        self.MODEL_ENGINE = 'aotengine'
        self.MODEL_ALIGN_CORNERS = True
        self.MODEL_ENCODER = 'mobilenetv2'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/mobilenet_v2-b0353104.pth'
        self.MODEL_ENCODER_DIM = [24, 32, 96, 1280]
        self.MODEL_ENCODER_EMBEDDING_DIM = 256
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True
        self.MODEL_LINEAR_Q = True
        self.MODEL_NORM_INP = True
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False
        self.MODEL_MAX_OBJ_NUM = 10
        self.MODEL_IGNORE_TOKEN = True
        self.MODEL_SELF_HEADS = 2
        self.MODEL_ATT_HEADS = 2
        self.MODEL_LSTT_NUM = 1
        self.MODEL_EPSILON = 1e-05
        self.TRAIN_LONG_TERM_MEM_GAP = 9999
        self.TEST_LONG_TERM_MEM_GAP = 9999
        self.LTMB_UPDATE_STRATEGY = 'ours'
        self.DEBUG_LTMB_UPDATE = False
        self.RECON_SCORE_WEIGHT = 1.0
        self.RMEM_RELEVANCE_EMA_BETA = 0.2
        self.RMEM_FRESHNESS_WEIGHT = 1.5
        self.RMEM_FRESHNESS_ADD = 8.0
