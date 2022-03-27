import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseNet, weights_init_classifier, weights_init_kaiming
# from .mgn import MGN
from .norm import Centering, Scale, OfflineBN


class CLSBaseline(BaseNet):

    num_reduction = 2048 # 256

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs):
        super().__init__(model_name, last_stride, pretrain_choice, model_path, **kwargs)
        # self.build_backbone(model_name, last_stride)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.scale = Scale(0)
        # self.reduction = nn.Linear(self.in_planes, self.num_reduction)

        # 不使用bnneck的话，resnet直接的输出全大于0的，这只占据整个特征空间中的一部分，但是fc是可以分布在全空间的
        if self.neck == 'no':
            self.bottleneck = nn.Sequential()
            self.classifier = nn.Linear(self.num_reduction, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.num_reduction)
            # self.bottleneck = OfflineBN(self.num_reduction)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.num_reduction, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'ct':
            self.bottleneck = Centering(self.num_reduction)
            # self.bottleneck = OfflineBN(self.num_reduction)
            self.classifier = nn.Linear(self.num_reduction, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        # with torch.no_grad():
        #     self.classifier.weight *= (1 / self.classifier.weight.norm(dim=-1, keepdim=True))

    def forward(self, x):
        feat = self.feature(x)
        # feat = self.reduction(feat)
        feat = self.bottleneck(feat)
        # feat = feat + self.feat_bias
        feat = self.scale(feat)

        if self.training:
            f_norm = feat.norm(dim=-1)
            w_norm = self.classifier.weight.norm(dim=-1)
            # cls_score = self.classifier(feat)
            # cos_sim = cls_score/f_norm[:, None]/w_norm
            cos_sim = F.linear(F.normalize(feat), F.normalize(self.classifier.weight))
            # cos_sim = self.classifier(feat)
            return cos_sim, f_norm, w_norm * 1    # 减少weight decay影响
        else:
            return feat


class CLSBaselineBN(CLSBaseline):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs):
        super().__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs)
        self.bn_f = nn.BatchNorm1d(self.num_reduction)
        # self.bn_f = OfflineBN(self.num_reduction)
        self.bn_f.bias.requires_grad_(False)  # no shift
        self.bn_f.apply(weights_init_kaiming)

    def feature(self, x):
        x = super().feature(x)
        x = self.bn_f(x)
        return x

class CLSBaselineFC(CLSBaseline):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs):
        super().__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs)
        self.bn_f = nn.BatchNorm1d(self.in_planes)
        self.fc_f = nn.Linear(self.in_planes, self.in_planes)
        self.dropout_f = nn.Dropout(p=0.2, inplace=True)
        nn.init.constant_(self.bn_f.weight, 0.5)

    def feature(self, x):
        x = super().feature(x)
        # x = self.bn_f(x)
        x = self.dropout_f(x)
        x = x.view(x.size(0), -1)
        x = self.fc_f(x)
        return x

class CLSBaselineCT(CLSBaseline):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, ct_type='no', **kwargs):
        super().__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, **kwargs)
        shift, scale = False, False
        if 'shift' in ct_type:
            shift=True
        if 'scale' in ct_type:
            scale=True
        self.centering = Centering(self.in_planes, shift=shift, scale=scale)

    def feature(self, x):
        x = super().feature(x)
        x = self.centering(x)
        return x


