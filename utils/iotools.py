# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import errno
import json
import os
import torch

import os.path as osp


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def resume_checkpoint(path, max_epoch, model='resnet50'):
    m_dict = dict()
    if os.path.exists(path):
        for f in os.listdir(path):      # resnet50_checkpoint_40.pt
            fn = f.split('.')
            if len(fn) == 2:
                fname, ext = fn
            else:
                continue
            if ext != 'pt':
                continue
            arch, _, epoch = fname.split('_')
            if arch != model:
                continue
            epoch = int(epoch)
            if epoch < max_epoch:
                m_dict[(epoch)] = f

    if len(m_dict) > 0:
        epoch = max(m_dict.keys())
        file = m_dict[epoch]
        state = torch.load(os.path.join(path, file), map_location='cpu')
        state['epoch'] = epoch
        return state
    else:
        return None

if __name__ == '__main__':
    resume_checkpoint('/data/code/deep-metric-learning/logs/Uncertain-fc/market1501/Exp-k5-uc-pair-none-10*12-s128-m0.0-[0.0, -0.4]-g1.0')
