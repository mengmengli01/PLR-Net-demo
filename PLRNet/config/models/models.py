from yacs.config import CfgNode as CN

MODELS = CN()

MODELS.NAME = "PLRNet"
MODELS.DEVICE = "cuda"
MODELS.HEAD_SIZE  = [[2]]
MODELS.OUT_FEATURE_CHANNELS = 256
MODELS.LOSS_WEIGHTS = CN(new_allowed=True)
