import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseNet, weights_init_classifier, weights_init_kaiming
# from .mgn import MGN
from .norm import Centering, Scale



class Baseline(BaseNet):

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs):
        super(Baseline, self).__init__(model_name, last_stride, pretrain_choice, model_path, **kwargs)
        # self.build_backbone(model_name, last_stride)
        # self.bn = nn.BatchNorm1d(self.in_planes)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.tscale = Scale(self.in_planes)
        self.tscale.requires_grad_(False)
        self.feat_bias.requires_grad_(False)

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.bottleneck = nn.Sequential()
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'no-bn':      # 训练也用bnneck后特征算triplet loss
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'scale':
            self.bottleneck = Scale(self.in_planes)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'norm':
            self.bottleneck = nn.Sequential()
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        # elif self.neck == 'bn-scale':
        #     self.bottleneck = nn.BatchNorm1d(self.in_planes)
        #     self.bottleneck.bias.requires_grad_(False)  # no shift
        #     # self.tscale.requires_grad_(True)
        #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        #     self.bottleneck.apply(weights_init_kaiming)
        #     self.classifier.apply(weights_init_classifier)
        # elif self.neck == 'sep-scale':
        #     self.bottleneck = Scale(self.in_planes)
        #     # self.tscale.requires_grad_(True)
        #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        #     self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        global_feat = self.feature(x)

        # if self.neck == 'no':
        #     feat = global_feat
        # elif self.neck == 'bnneck':
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        # feat = feat + self.feat_bias
        # global_feat = global_feat + self.feat_bias
        # global_feat = self.tscale(global_feat)

        if self.training:
            if self.neck == 'norm':
                cls_score = F.linear(F.normalize(feat), F.normalize(self.classifier.weight))
            else:
                cls_score = self.classifier(feat)
            if self.neck == 'no-bn':        # bn 后特征算tri
                global_feat = feat
            # re-ensamble feature norms
            # global_feat = F.normalize(global_feat) * feat.norm(dim=-1, keepdim=True)  # bn前的方向+bn后的norm
            # global_feat = F.normalize(feat) * global_feat.norm(dim=-1, keepdim=True)   # bn前的norm+bn后的方向
            return cls_score, global_feat[:, ::1]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                feat = F.normalize(feat) * global_feat.norm(dim=-1, keepdim=True)
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat


class BaselineFC(Baseline):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs):
        super().__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs)
        self.bn_f = nn.BatchNorm1d(self.in_planes)
        self.fc_f = nn.Linear(self.in_planes, self.in_planes)
        self.dropout_f = nn.Dropout(p=0.2, inplace=True)
        # self.feat_bias.requires_grad_(True)

    def feature(self, x):
        x = super().feature(x)
        # x = self.bn_f(x)
        x = self.dropout_f(x)
        x = x.view(x.size(0), -1)
        x = self.fc_f(x)
        return x


class BaselineBN(Baseline):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs):
        super().__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs)
        self.bn_f = nn.BatchNorm1d(self.in_planes)
        self.bn_f.bias.requires_grad_(False)  # no shift
        self.bn_f.apply(weights_init_kaiming)
        self.feat_bias.requires_grad_(True)

    def feature(self, x):
        x = super().feature(x)
        # x1 = self.bn_f(x)
        # return F.normalize(x1) * x.norm(dim=-1, keepdim=True)
        x = self.bn_f(x)
        return x

class BaselineBN2d(Baseline):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, bn_init=1, **kwargs):
        super().__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs)
        self.bn_f = nn.BatchNorm2d(self.in_planes)
        nn.init.constant_(self.bn_f.weight, bn_init)
        self.bn_f.bias.requires_grad_(False)  # no shift
        self.bn_f.apply(weights_init_kaiming)
        self.feat_bias.requires_grad_(True)

    def feature(self, x):
        featmap = self.base(x)
        # x1 = self.bn_f(x)
        # return F.normalize(x1) * x.norm(dim=-1, keepdim=True)
        x = self.bn_f(featmap)
        global_feat = self.gap(x)  # (b, 2048, 1, 1)  [:, :self.in_planes]
        # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        return global_feat.view(global_feat.shape[0], -1)

class BaselineCT(Baseline):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, ct_type='no', **kwargs):
        super().__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs)

        # self.feat_bias.requires_grad_(True)
        shift, scale = False, False
        if 'shift' in ct_type:
            shift=True
        if 'scale' in ct_type:
            scale=True
        self.centering = Centering(self.in_planes, shift=shift, scale=scale)

    def feature(self, x):
        x = super().feature(x)
        x1 = self.centering(x)
        return x1
        # return F.normalize(x1) * x.norm(dim=-1, keepdim=True)

