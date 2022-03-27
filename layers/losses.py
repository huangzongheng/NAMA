"""
Open Source Software Licensed under the Apache License Version 2.0:
--------------------------------------------------------------------
The below software in this distribution may have been modified by Megvii Inc. (“Megvii Modifications”). All Megvii Modifications
are Copyright (C) 2014-2020 Megvii Inc. All rights reserved.

Copyright (c) 2020, Megvii Inc. Holding Limited
"""
from __future__ import absolute_import
import sys

import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['DeepSupervision', 'CrossEntropyLabelSmooth', 'TripletLoss',
           'ArcFaceLoss', 'CosineFaceLoss']


def LogSumExp(score, mask):
    max_score = score.max()
    max_score = max_score.unsqueeze(0).unsqueeze(1).expand_as(score)
    score = score - max_score * (1-mask)   # elimintate the scores which are of none use
    max_score, _ = score.max(1)
    max_score_reduce = max_score.unsqueeze(1).expand_as(score)
    score = score - max_score_reduce
    return max_score + ((score.exp() * mask).sum(1)).log()


def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='euclidean', use_gpu=True):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'consine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.use_gpu = use_gpu
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'consine':
            fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
            l2norm = inputs.div(fnorm.expand_as(inputs))
            dist = - torch.mm(l2norm, l2norm.t())

        if self.use_gpu: targets = targets.cuda()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class ArcFaceLoss(nn.Module):
    def __init__(self, m=0.3, s=64, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.m = m
        self.s = s
        self.easy_margin = easy_margin

    def forward(self, input, target):

        # make a one-hot index
        index = input.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        cos_t = input[index]
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        cos_t_add_m = cos_t * cos_m  - sin_t * sin_m

        if self.easy_margin:
            cond = F.relu(cos_t)
            keep = cos_t
        else:
            cond_v = cos_t - math.cos(math.pi - self.m)
            cond = F.relu(cond_v)
            keep = cos_t - math.sin(math.pi - self.m) * self.m

        cos_t_add_m = torch.where(cond.byte(), cos_t_add_m, keep)

        output = input * 1.0 #size=(B,Classnum)
        output[index] = cos_t_add_m
        output = self.s * output

        return F.cross_entropy(output, target)


class CosineFaceLoss(nn.Module):
    def __init__(self, m=0.3, s=64):
        super(CosineFaceLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pw = None
        self.simi_nw = None

    def forward(self, input, target):


        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        self.simi_pw = torch.masked_select(input, one_hot.bool()).mean()
        self.simi_nw = LogSumExp(input * self.s, 1 - one_hot).mean() / self.s

        output = self.s * (input - one_hot * self.m)

        return F.cross_entropy(output, target)

if __name__ == '__main__':
    pass


class CircleLoss(nn.Module):
    def __init__(self, m=0.25, s=96):
        super(CircleLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pw = None
        self.simi_nw = None

    def forward(self, input, target):

        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)

        self.simi_pw = torch.masked_select(input, one_hot.bool()).mean()
        self.simi_nw = LogSumExp(input, 1 - one_hot).mean()

        pos_scale = self.s * F.relu(1 + self.m - input).detach()
        neg_scale = self.s * F.relu(input + self.m).detach()
        scale_matrix = pos_scale * one_hot + neg_scale * (1 - one_hot)

        score = (input - (1 - self.m) * one_hot - self.m * (1 - one_hot)) * scale_matrix

        return F.cross_entropy(score, target)

if __name__ == '__main__':
    pass


class CosfacePairwiseLoss(nn.Module):
    def __init__(self, m=0.35, s=16):
        super(CosfacePairwiseLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pos = None
        self.simi_neg = None

    def forward(self, input, target):

        n = input.size(0)
        target = target.cuda()
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        mask = mask.float()
        mask_self = torch.FloatTensor(np.eye(n)).cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        simi = input.mm(input.t())
        self.simi_pos = LogSumExp(- simi * self.s, mask_pos).mean() / (- self.s)
        self.simi_neg = LogSumExp(simi * self.s, mask_neg).mean() / self.s
        simi = (simi - self.m * mask) * self.s

        # '''
        pos_LSE_cmp = LogSumExp(- simi, mask_pos)
        neg_LSE_cmp = LogSumExp(simi, mask_neg)

        loss_cmp = F.softplus(pos_LSE_cmp + neg_LSE_cmp)
        # '''

        '''
        mask_pos, mask_neg = mask_pos.bool(), mask_neg.bool()
        pos_pairs = torch.masked_select(simi, mask_pos).reshape(n, -1)
        neg_pairs = torch.masked_select(simi, mask_neg).reshape(n, -1)
        pos_LSE = torch.logsumexp(- pos_pairs, 1)
        neg_LSE = torch.logsumexp(neg_pairs, 1)
        loss = F.softplus(pos_LSE + neg_LSE)
        '''

        return loss_cmp.mean()


if __name__ == '__main__':
    pass


class CirclePairwiseLoss(nn.Module):
    def __init__(self, m=0.25, s=16):
        super(CirclePairwiseLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pos = None
        self.simi_neg = None

    def forward(self, input, target, center=(0,1)):

        n = input.size(0)
        target = target.cuda()
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        mask = mask.int()
        mask_self = torch.FloatTensor(np.eye(n)).cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        simi = input.mm(input.t())

        pos_scale = self.s * F.relu(center[1] + self.m - simi).detach()
        neg_scale = self.s * F.relu(simi + self.m - center[0]).detach()
        scale_matrix = pos_scale * mask_pos + neg_scale * neg_scale

        score = (center[1] - self.m - simi) * mask_pos + (simi - self.m - center[0]) * mask_neg
        score = score * scale_matrix
        self.simi_pos, self.simi_neg = pos_LSE_cmp.mean(), neg_LSE_cmp.mean()
        # '''
        pos_LSE_cmp = LogSumExp(score, mask_pos)
        neg_LSE_cmp = LogSumExp(score, mask_neg)

        loss_cmp = F.softplus(pos_LSE_cmp + neg_LSE_cmp)
        # '''

        '''
        mask_pos, mask_neg = mask_pos.bool(), mask_neg.bool()
        pos_pairs = torch.masked_select(simi, mask_pos).reshape(n, -1)
        neg_pairs = torch.masked_select(simi, mask_neg).reshape(n, -1)
        pos_LSE = torch.logsumexp(- pos_pairs, 1)
        neg_LSE = torch.logsumexp(neg_pairs, 1)
        loss = F.softplus(pos_LSE + neg_LSE)
        '''

        return loss_cmp.mean()


if __name__ == '__main__':
    pass


class LSEContrastiveLoss(nn.Module):
    def __init__(self, m=0.35, s=16, detach=False):
        super(LSEContrastiveLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pos = None
        self.simi_neg = None
        self.detach = detach

    def forward(self, input, target):

        n = input.size(0)
        target = target.cuda()
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        mask = mask.int()
        mask_self = torch.FloatTensor(np.eye(n)).cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        simi = input.mm(input.t())
        if self.detach:
            self.simi_pos = (LogSumExp(- simi * self.s, mask_pos).mean() / (- self.s)).mean().detach()
            self.simi_neg = (LogSumExp(simi * self.s, mask_neg).mean() / self.s).mean().detach()
        else:
            self.simi_pos = (LogSumExp(- simi * self.s, mask_pos).mean() / (- self.s)).mean()
            self.simi_neg = (LogSumExp(simi * self.s, mask_neg).mean() / self.s).mean()

        pos_LSE = LogSumExp((self.simi_pos - simi) * self.s, mask_pos)
        neg_LSE = LogSumExp((simi - self.simi_neg) * self.s, mask_neg)

        # '''

        loss_cmp = F.softplus(pos_LSE) + F.softplus(neg_LSE)
        # '''

        '''
        mask_pos, mask_neg = mask_pos.bool(), mask_neg.bool()
        pos_pairs = torch.masked_select(simi, mask_pos).reshape(n, -1)
        neg_pairs = torch.masked_select(simi, mask_neg).reshape(n, -1)
        pos_LSE = torch.logsumexp(- pos_pairs, 1)
        neg_LSE = torch.logsumexp(neg_pairs, 1)
        loss = F.softplus(pos_LSE + neg_LSE)
        '''

        return loss_cmp.mean()


if __name__ == '__main__':
    pass


class LSEContrastiveRowLoss(nn.Module):
    def __init__(self, m=0.35, s=16, detach=False):
        super(LSEContrastiveRowLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pos = None
        self.simi_neg = None
        self.detach = detach

    def forward(self, input, target):

        n = input.size(0)
        target = target.cuda()
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        mask = mask.int()
        mask_self = torch.FloatTensor(np.eye(n)).cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        simi = input.mm(input.t())
        if self.detach:
            simi_pos = (LogSumExp(- simi * self.s, mask_pos) / (- self.s)).detach()
            simi_neg = (LogSumExp(simi * self.s, mask_neg) / self.s).detach()
        else:
            simi_pos = (LogSumExp(- simi * self.s, mask_pos) / (- self.s))
            simi_neg = (LogSumExp(simi * self.s, mask_neg) / self.s)

        self.simi_pos = simi_pos.mean()
        self.simi_neg = simi_neg.mean()

        pos_LSE = LogSumExp((simi_pos - simi) * self.s, mask_pos)
        neg_LSE = LogSumExp((simi - simi_neg) * self.s, mask_neg)

        # '''

        loss_cmp = F.softplus(pos_LSE) + F.softplus(neg_LSE)
        # loss_cmp = F.softplus(pos_LSE + neg_LSE)
        # '''

        '''
        mask_pos, mask_neg = mask_pos.bool(), mask_neg.bool()
        pos_pairs = torch.masked_select(simi, mask_pos).reshape(n, -1)
        neg_pairs = torch.masked_select(simi, mask_neg).reshape(n, -1)
        pos_LSE = torch.logsumexp(- pos_pairs, 1)
        neg_LSE = torch.logsumexp(neg_pairs, 1)
        loss = F.softplus(pos_LSE + neg_LSE)
        '''

        return loss_cmp.mean()


if __name__ == '__main__':
    pass


class LSECircleRowLoss(nn.Module):
    def __init__(self, m=0.15, s=16, detach=False):
        super(LSECircleRowLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pos = None
        self.simi_neg = None
        self.detach = detach

    def forward(self, input, target):

        n = input.size(0)
        target = target.cuda()
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        mask = mask.int()
        mask_self = torch.FloatTensor(np.eye(n)).cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        simi = input.mm(input.t())
        pos_scale = self.s * F.relu(1 + self.m - simi).detach()
        neg_scale = self.s * F.relu(simi + self.m).detach()
        scale_matrix = pos_scale * mask_pos + neg_scale * neg_scale
        if self.detach:
            simi_pos = LogSumExp(- simi * scale_matrix, mask_pos).detach()
            simi_neg = LogSumExp(simi * scale_matrix, mask_neg).detach()
        else:
            simi_pos = (LogSumExp(- simi * scale_matrix, mask_pos))
            simi_neg = (LogSumExp(simi * scale_matrix, mask_neg))

        self.simi_pos = simi_pos.mean()
        self.simi_neg = simi_neg.mean()

        pos_LSE = LogSumExp(-simi_pos - simi * scale_matrix, mask_pos)
        neg_LSE = LogSumExp(simi * scale_matrix - simi_neg, mask_neg)

        # '''

        loss_cmp = F.softplus(pos_LSE) + F.softplus(neg_LSE)
        # '''

        '''
        mask_pos, mask_neg = mask_pos.bool(), mask_neg.bool()
        pos_pairs = torch.masked_select(simi, mask_pos).reshape(n, -1)
        neg_pairs = torch.masked_select(simi, mask_neg).reshape(n, -1)
        pos_LSE = torch.logsumexp(- pos_pairs, 1)
        neg_LSE = torch.logsumexp(neg_pairs, 1)
        loss = F.softplus(pos_LSE + neg_LSE)
        '''

        return loss_cmp.mean()


if __name__ == '__main__':
    pass


class LSEContrastiveRowPlusLoss(nn.Module):
    def __init__(self, m=0.35, s_scatter=16, s_pairwise=32, detach=False):
        super(LSEContrastiveRowPlusLoss, self).__init__()
        self.m = m
        self.s_scatter = s_scatter
        self.s_pairwise = s_pairwise
        self.simi_pos = None
        self.simi_neg = None
        self.detach = detach

    def forward(self, input, target):

        n = input.size(0)
        target = target.cuda()
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        mask = mask.int()
        mask_self = torch.FloatTensor(np.eye(n)).cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        simi = input.mm(input.t())
        if self.detach:
            simi_pos = (LogSumExp(- simi * self.s_scatter, mask_pos) / (- self.s_scatter)).detach()
            simi_neg = (LogSumExp(simi * self.s_scatter, mask_neg) / self.s_scatter).detach()
        else:
            simi_pos = (LogSumExp(- simi * self.s_scatter, mask_pos) / (- self.s_scatter))
            simi_neg = (LogSumExp(simi * self.s_scatter, mask_neg) / self.s_scatter)

        self.simi_pos = simi_pos.mean()
        self.simi_neg = simi_neg.mean()

        pos_LSE = LogSumExp((simi_pos - simi) * self.s_scatter, mask_pos)
        neg_LSE = LogSumExp((simi - simi_neg) * self.s_scatter, mask_neg)

        # '''

        loss_cmp = F.softplus(pos_LSE) + F.softplus(neg_LSE)
        # loss_cmp = F.softplus(pos_LSE + neg_LSE)
        # '''

        score = (simi - self.m * mask) * self.s_pairwise

        # '''
        pos_LSE_cmp = LogSumExp(- score, mask_pos)
        neg_LSE_cmp = LogSumExp(score, mask_neg)

        loss_pairwise = F.softplus(pos_LSE_cmp + neg_LSE_cmp)

        return loss_cmp.mean(), loss_pairwise.mean()


if __name__ == '__main__':
    pass


class LSEContrastiveCLSLoss(nn.Module):
    def __init__(self, m=0.35, s=16, detach=True):
        super(LSEContrastiveCLSLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pos = None
        self.simi_neg = None
        self.detach = detach

    def forward(self, input, target):

        n = input.size(0)
        s_scatter = self.s / 1
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        simi = input.reshape(n//1, -1)
        mask_pos = one_hot.reshape(n//1, -1)
        mask_neg = 1 - mask_pos

        '''
        if self.detach:
            simi_pos = (LogSumExp(- simi * s_scatter, mask_pos).mean() / (- s_scatter)).mean().detach()
            simi_neg = (LogSumExp(simi * s_scatter, mask_neg).mean() / s_scatter).mean().detach()
        else:
            simi_pos = (LogSumExp(- simi * s_scatter, mask_pos).mean() / (- s_scatter)).mean()
            simi_neg = (LogSumExp(simi * s_scatter, mask_neg).mean() / s_scatter).mean()
        '''

        if self.detach:
            simi_pos = (LogSumExp(- simi * self.s, mask_pos) / (- self.s)).detach()
            simi_neg = (LogSumExp(simi * self.s, mask_neg) / self.s).detach()
        else:
            simi_pos = (LogSumExp(- simi * self.s, mask_pos) / (- self.s))
            simi_neg = (LogSumExp(simi * self.s, mask_neg) / self.s)

        pos_LSE = LogSumExp((simi_pos.unsqueeze(1).expand_as(simi) - simi) * s_scatter, mask_pos)
        neg_LSE = LogSumExp((simi - simi_neg.unsqueeze(1).expand_as(simi)) * s_scatter, mask_neg)

        self.simi_pos = simi_pos.mean()
        self.simi_neg = simi_neg.mean()

        loss_cmp = F.softplus(pos_LSE) + F.softplus(neg_LSE)
        loss_cmp = F.softplus(pos_LSE + neg_LSE)

        output = self.s * (input - one_hot * self.m)
        return 0 * loss_cmp.mean() + F.cross_entropy(output, target)



