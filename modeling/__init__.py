# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

# from .baseline import Baseline
from .tri_base import Baseline, BaselineFC, BaselineBN, BaselineCT, MGN_BN
from .cls_base import CLSBaseline, CLSBaselineFC, CLSBaselineCT, CLSBaselineBN, CLS_MGN
from .uc_base import UcCls, UcTri, UcTriBN, UcTriCT, UcTriBN2d
# from .baseline_attr import BaselineAttr
# from .baseline_C2F import BaselineC2F
# from .qg_net import QG_Net
# from .PCB import PCB
# from .region_net import RegionNet


def build_model(cfg, num_classes, num_attr=0, local_layer=2, net=Baseline):
    if net == 'baseline':   # 标准bnneck结构
        net = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                       cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    elif net == 'clsbaseline':
        net = CLSBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                       cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    elif net == 'clsuc':
        net = UcCls(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                       cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, fbase=cfg.MODEL.F_BASE)
    elif net == 'uc':   # bnneck网络加入不确定性模块
        net = UcTri(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                       cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, fbase=cfg.MODEL.F_BASE)
    else:
        net = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                       cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)

    return net
