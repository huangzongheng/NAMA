# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from .backbones.resnet import ResNet, BasicBlock, Bottleneck, BottleneckHomo
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .backbones.resnet_affine import AffineResNet, AffineBottleneck
from .base import BaseNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(BaseNet):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs):
        super(Baseline, self).__init__(model_name, last_stride, pretrain_choice, model_path)
        # self.build_backbone(model_name, last_stride)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)


    def forward(self, x, aff=None):
        if aff is not None and isinstance(self.base, AffineResNet):
            featmap = self.base(x, aff)  # (b, 2048, h, w)
        else:
            featmap = self.base(x)
        if isinstance(featmap, tuple):
            global_feat, affine = featmap
        else:
            global_feat = featmap
            affine = None
        global_feat = self.gap(global_feat)  # (b, 2048, 1, 1)  [:, :self.in_planes]
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            # re-ensamble feature norms
            # global_feat = F.normalize(global_feat) * feat.norm(dim=-1, keepdim=True)  # bn前的方向+bn后的norm

            global_feat = F.normalize(feat) * global_feat.norm(dim=-1, keepdim=True)   # bn前的norm+bn后的方向
            if affine is not None:
                global_feat = (global_feat, affine)
            return cls_score, global_feat[:, ::1]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                feat = F.normalize(feat) * global_feat.norm(dim=-1, keepdim=True)
                return feat[:, ::1]
            else:
                # print("Test with feature before BN")
                return global_feat[:, ::1]

