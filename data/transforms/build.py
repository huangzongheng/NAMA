# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from collections.abc import Sequence, Iterable
from .transforms import RandomErasing, RandomVerticalCropCont, ListToTensor, AutoResize, RandomVerticalErase, \
    AddMask, DownCropResize, AddPosMask, ResumePosMask

# T.RandomAffine

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            # AutoResize(cfg.INPUT.SIZE_TEST, cfg.INPUT.VE_RATE),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
            # RandomVerticalErase(cfg.INPUT.VE_RATE, cfg.INPUT.VE_MODE),
        ])
    else:
        transform = T.Compose([
            # T.Resize((5*cfg.INPUT.SIZE_TEST[0]//4, cfg.INPUT.SIZE_TEST[1])),
            # T.Pad((0, cfg.INPUT.SIZE_TEST[0]//4, 0, 0)),
            # T.CenterCrop(cfg.INPUT.SIZE_TEST),
            # T.Resize(cfg.INPUT.SIZE_TEST),
            AutoResize(cfg.INPUT.SIZE_TEST, cfg.TEST.RESIZE_MODE),
            T.ToTensor(),
            normalize_transform,
            # DownCropResize(cfg.TEST.CROP_RATIO)
            # AddMask(cfg.INPUT.SIZE_TEST, cfg.TEST.RESIZE_MODE)
        ])

    return transform


def build_ema_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(*cfg.INPUT.JITTER),        # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            T.RandomAffine(*cfg.INPUT.AFFINE, resample=3),
            # AutoResize(cfg.INPUT.SIZE_TEST, cfg.INPUT.VE_RATE),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
            # RandomVerticalErase(cfg.INPUT.VE_RATE, cfg.INPUT.VE_MODE),
        ])
    else:
        transform = T.Compose([
            # T.Resize((5*cfg.INPUT.SIZE_TEST[0]//4, cfg.INPUT.SIZE_TEST[1])),
            # T.Pad((0, cfg.INPUT.SIZE_TEST[0]//4, 0, 0)),
            # T.CenterCrop(cfg.INPUT.SIZE_TEST),
            T.Resize(cfg.INPUT.SIZE_TEST),
            # AutoResize(cfg.INPUT.SIZE_TEST, cfg.TEST.RESIZE_MODE),
            T.ToTensor(),
            normalize_transform,
            # DownCropResize(cfg.TEST.CROP_RATIO)
            # AddMask(cfg.INPUT.SIZE_TEST, cfg.TEST.RESIZE_MODE)
        ])

    return transform


def build_qg_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomVerticalCropCont(*cfg.INPUT.SIZE_TRAIN),
            ListToTensor(normalize_transform),
            # normalize_transform,
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize((5*cfg.INPUT.SIZE_TEST[0]//4, cfg.INPUT.SIZE_TEST[1])),
            T.Pad((0, cfg.INPUT.SIZE_TEST[0]//4, 0, 0)),
            T.CenterCrop(cfg.INPUT.SIZE_TEST),
            # T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

# todo:将原始图片空间坐标随图片一起变化，以记录空间位置信息
def build_pixel_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            # T.ColorJitter(*cfg.INPUT.JITTER),        # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            # T.RandomAffine(*cfg.INPUT.AFFINE, resample=3),
            # AutoResize(cfg.INPUT.SIZE_TEST, cfg.INPUT.VE_RATE),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            AddPosMask(20),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),     #
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=[0, 0, 0]),   # 不擦坐标
            ResumePosMask(20)
        ])
        transform1 = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            # T.ColorJitter(0.1, 0.1, 0.1, 0.1),        # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            T.ToTensor(),
            normalize_transform,
            AddPosMask(20),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad((20,40)),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            ResumePosMask(20)
        ])
        # 保证两边用相同的flip
        pre_transform = T.Compose([
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        ])
        transform = [transform, transform1, pre_transform]
    else:
        transform = T.Compose([
            # T.Resize((5*cfg.INPUT.SIZE_TEST[0]//4, cfg.INPUT.SIZE_TEST[1])),
            # T.Pad((0, cfg.INPUT.SIZE_TEST[0]//4, 0, 0)),
            # T.CenterCrop(cfg.INPUT.SIZE_TEST),
            T.Resize(cfg.INPUT.SIZE_TEST),
            # AutoResize(cfg.INPUT.SIZE_TEST, cfg.TEST.RESIZE_MODE),
            T.ToTensor(),
            normalize_transform,
            # DownCropResize(cfg.TEST.CROP_RATIO)
            # AddMask(cfg.INPUT.SIZE_TEST, cfg.TEST.RESIZE_MODE)
        ])

    return transform

