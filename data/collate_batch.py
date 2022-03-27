# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, _, fname, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # print(pids)
    # print(fname)
    return torch.stack(imgs, dim=0), pids

def train_attr_collate_fn(batch):
    imgs, pids, cids, fname, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    cids = torch.tensor(cids, dtype=torch.int64)
    # print(pids)
    # print(fname)
    return torch.stack(imgs, dim=0), pids, cids     # attr_label

def train_ema_collate_fn(batch):
    imgs, pids, _, fname, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # print(pids)
    # print(fname)
    return torch.stack(imgs, dim=0), pids

def train_qg_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    # imgs, imgt, ratio, start = imgs
    pids = torch.tensor(pids, dtype=torch.int64)

    # return torch.stack(imgs, dim=0), pids
    return [torch.stack(i, dim=0) for i in zip(*imgs)], pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
