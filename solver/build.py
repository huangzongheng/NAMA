# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
from .ranger import Ranger


def make_optimizer(cfg, model):
    # fix backbone
    for key, value in model.named_parameters():
        if cfg.SOLVER.TRAIN_MODE == 'all':
            break
        elif cfg.SOLVER.TRAIN_MODE == 'base':
            if 'affine' in key:
                value.requires_grad_(False)
        elif cfg.SOLVER.TRAIN_MODE == 'affine':
            if 'affine' not in key:
                value.requires_grad_(False)
        else:
            break
        # if not value.requires_grad:
        #     continue
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if "uc_k" in key:
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_POLY
        if "neck" in key or "classifier" in key:
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NECK
        if "bn_f" in key:
            weight_decay = 0 # cfg.SOLVER.WEIGHT_DECAY_NECK
        # elif "head" in key:
        #     lr = cfg.SOLVER.BASE_LR * 10
        #     weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        # elif "attention.1" in key:
        #     lr = cfg.SOLVER.BASE_LR * 10
        #     weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center

def make_optimizer_region(cfg, model):
    # fix backbone
    for key, value in model.named_parameters():
        if cfg.SOLVER.TRAIN_MODE == 'all':
            # for fix_layer in cfg.SOLVER.FIXED_LAYER:        # freeze certain layers
            #     if fix_layer in key:
            #         value.requires_grad_(False)
            break
        # elif cfg.SOLVER.TRAIN_MODE == 'base':
        #     if 'head' in key:
        #         value.requires_grad_(False)
        elif cfg.SOLVER.TRAIN_MODE == 'head':
            if 'head' not in key:
                value.requires_grad_(False)
        else:
            break
        # if not value.requires_grad:
        #     continue
    params = []
    for key, value in model.named_parameters():
        # if not value.requires_grad:
        #     continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif "head" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.HEAD_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer


def freeze_specified_layers(model, layers):
    for key, value in model.named_parameters():
        for fix_layer in layers:        # freeze certain layers
            if fix_layer in key:
                value.requires_grad_(False)
