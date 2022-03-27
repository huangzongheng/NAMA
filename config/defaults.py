from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# arch of backbone
_C.MODEL.ARCH = 'baseline'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# center neck type  no/shift/scale/shift-scale
_C.MODEL.CT_TYPE = 'no'
# function base for uc calculaton: tlr lgd
_C.MODEL.F_BASE = 'tlr'
# use which layer as local feature outputs
_C.MODEL.LOCAL_LAYER = 3
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.CLS_LOSS_TYPE = 'softmax'
# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'

# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# Value of vertical erase ratio
_C.INPUT.VE_RATE = 1.0
# mode of vertical erase ratio :pad resize
_C.INPUT.VE_MODE = "pad"
# color jitter para: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
_C.INPUT.JITTER = [0, 0, 0, 0]
# random affine para: degrees=10, translate=(0.1, 0.05), scale=(0.75,1.33), shear=0.1,
_C.INPUT.AFFINE = [0, None, None, None, ]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./datasets')
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.COMBINE_ALL = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# Number similar classes for one batch
_C.DATALOADER.K = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# optimize level for apex
_C.SOLVER.OPT_LEVEL = "O1"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# fix certain layers in the first few epochs
_C.SOLVER.FIXED_LAYER = []
# num epochs training with fixed layers
_C.SOLVER.FIX_EPOCHS = 0
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 2
# Factor of learning bias
_C.SOLVER.HEAD_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# use adaptive weight for pixel moco
_C.SOLVER.ADP_WEIGHT = False
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# margin norm for mag-tri
_C.SOLVER.MARGIN_LU = [0.0, 0.0]
_C.SOLVER.NORM_LU = [10, 110]
# function type of reg term
_C.SOLVER.REG_TYPE = 'exp_3'
# Balanced weight of reg term
_C.SOLVER.MAG_G = 1.0
# Balanced weight of triplet loss
_C.SOLVER.TRI_LOSS_WEIGHT = 1.0
# min weight of triplet loss at start
_C.SOLVER.TW_MIN = 0.0
# type of triplet loss. eg.'tri' 'r-tri'
_C.SOLVER.TRI_TYPE = 'tri'
# training mode "base" "att" "all" "affine"
_C.SOLVER.TRAIN_MODE = "base"
# scale for pairwise cosface
_C.SOLVER.S = 256
# scale for cosface cls
_C.SOLVER.CS = 16
# update rate of gr-tri
_C.SOLVER.ALPHA = 0.2
# update rate of gr-tri
_C.SOLVER.BETA = 0.9
# scale factor for hard triplet mining
_C.SOLVER.TRIGAMMA = 0
# scale factor for uncertainty regularization
_C.SOLVER.ATT_LAMBDA = 0.1
# hard example selection rate of ghr-tri
_C.SOLVER.P = 1.0
# very hard example filter sigma of ghr-tri
_C.SOLVER.SIGMA = 2.0
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# momentum of moco encoder
_C.SOLVER.MOCO_MOMENTUM = 0.999
# loss function of moco : nce, contrastive
_C.SOLVER.MOCO_LOSS = "nce"
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1
# region loss
_C.SOLVER.REGION_N = 5
_C.SOLVER.REGION_T = 0.2
_C.SOLVER.REGION_M = 0.1
_C.SOLVER.REGION_S = 16


# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.
_C.SOLVER.WEIGHT_DECAY_POLY = 0.05
_C.SOLVER.WEIGHT_DECAY_NECK = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (30, 55)

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 120
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# visualize top k ranking list, 0 for disable
_C.TEST.VISRANK = 0
# use norm as uncertainty for evaluation
_C.TEST.EVAL_NORM = False
# papra k for using norm as uncertainty for evaluation
_C.TEST.NORM_K = 0.0
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# how to fit given test image to certain size: pad, crop, resize
_C.TEST.RESIZE_MODE = 'resize'
# how much ratio of img to be preserved
_C.TEST.CROP_RATIO = 1.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./logs"
# seed for rng
_C.SEED = 42
