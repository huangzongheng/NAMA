# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth, NoneCLS, NoneTri, RelativeTripletLoss, MagRelativeTripletLoss
    # BaseRelativeTripletLoss, TripletPosLoss, TightRelativeTripletLoss, ClassRelativeTripletLoss, FocalRelativeTripletLoss, ADMarginRelativeTripletLoss
from .center_loss import CenterLoss
from .circle_loss import CircleLoss
# from .sim_loss import SimLoss, resize_score
from .cosface import CosfacePairwiseLoss, CosFaceLoss, ArcFaceLoss, ArcFacePairwiseLoss, ProjCLSLoss, NormFace
# from .mutilabel_loss import att_loss
# from .pix_moco_trainer import PixMoco
# from .magtri import MagTripletLoss# , UCTripletLoss
from .UCloss import UCCLSLoss, UCPairwiseLoss, UCTripletLoss, UCArcTripletLoss, UCNormedCLSLoss, UCNormedPairwiseLoss, \
    UCXentLoss, UCShiftCLSLoss, UCShiftPairwiseLoss, UCShiftTripletLoss, UCShiftArcTripletLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    triplet = make_metric_loss(cfg, num_classes)
    classification = make_cls_loss(cfg, num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    # elif sampler == 'softmax_triplet'
    else:
        if 'cls' in cfg.MODEL.ARCH:
            def loss_func(score, f_norm, w_norm, uc=None, target=None):
                # if not hasattr(loss_func, 'weight_t'):
                #     loss_func.weight_t = 0
                # if not hasattr(loss_func, 'triplet'):
                #     loss_func.triplet = triplet
                if uc is None:
                    l_cls = classification(score, f_norm, w_norm, target)
                else:
                    l_cls = classification(score, f_norm, w_norm, uc, target)
                # l_tri = triplet(feat, target)[0]
                return l_cls, 0 * l_cls.detach()
            triplet = classification
        else:
            def loss_func(score, feat, uc=None, target=None, weight=None):
                # if not hasattr(loss_func, 'weight_t'):
                #     loss_func.weight_t = cfg.SOLVER.TRI_LOSS_WEIGHT
                # if not hasattr(loss_func, 'triplet'):
                #     loss_func.triplet = triplet
                l_cls = classification(score, target)
                if uc is None or ('uc' not in cfg.MODEL.METRIC_LOSS_TYPE):  #  cfg.MODEL.ARCH
                    l_tri = triplet(feat, target)[0] * loss_func.weight_t # cfg.SOLVER.TRI_LOSS_WEIGHT
                else:
                    l_tri = triplet(feat, uc, target)[0] * loss_func.weight_t # cfg.SOLVER.TRI_LOSS_WEIGHT

                return l_cls + l_tri, l_tri
    loss_func.triplet = triplet
    loss_func.weight_t = cfg.SOLVER.TRI_LOSS_WEIGHT
    return loss_func


def make_metric_loss(cfg, num_classes):


    if cfg.MODEL.METRIC_LOSS_TYPE == 'tri':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'tri-n':
        triplet = TripletLoss(cfg.SOLVER.MARGIN, True)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'circle':
        triplet = CircleLoss()
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cosface':
        triplet = CosfacePairwiseLoss(m=cfg.SOLVER.MARGIN, s=cfg.SOLVER.S)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'arcface':
        triplet = ArcFacePairwiseLoss(m=cfg.SOLVER.MARGIN, s=cfg.SOLVER.S)

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-atri':
        triplet = UCArcTripletLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                   lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                                   reg_type=cfg.SOLVER.REG_TYPE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-tri':
        triplet = UCTripletLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                                reg_type=cfg.SOLVER.REG_TYPE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-pair':
        triplet = UCPairwiseLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                 lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                                 s=cfg.SOLVER.S, reg_type=cfg.SOLVER.REG_TYPE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-pair-free':
        if 'uc' in cfg.MODEL.ARCH:
            triplet = UCShiftPairwiseLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                           lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1],
                                           lambda_g=0, normalize=False,
                                           s=cfg.SOLVER.S, reg_type=cfg.SOLVER.REG_TYPE)
        else:
            raise NotImplementedError
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-pair-n':

        if 'uc' in cfg.MODEL.ARCH:
            triplet = UCShiftPairwiseLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                           lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1],
                                           lambda_g=cfg.SOLVER.MAG_G,
                                           s=cfg.SOLVER.S, reg_type=cfg.SOLVER.REG_TYPE)
        else:
            triplet = UCNormedPairwiseLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                           lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1],
                                           lambda_g=cfg.SOLVER.MAG_G,
                                           s=cfg.SOLVER.S, reg_type=cfg.SOLVER.REG_TYPE)

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-tri-n':
        if 'uc' in cfg.MODEL.ARCH:
            triplet = UCShiftTripletLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                           lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1],
                                           lambda_g=cfg.SOLVER.MAG_G,
                                           s=cfg.SOLVER.S, reg_type=cfg.SOLVER.REG_TYPE)
        else:
            raise NotImplementedError
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-tri-hard':
        if 'uc' in cfg.MODEL.ARCH:
            triplet = UCShiftTripletLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                           lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1],
                                           lambda_g=cfg.SOLVER.MAG_G,
                                           s=cfg.SOLVER.S, reg_type=cfg.SOLVER.REG_TYPE, soft_mining=False)
        else:
            raise NotImplementedError
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-tri-dpn':        # 只回传anchor的自适应margin梯度
        if 'uc' in cfg.MODEL.ARCH:
            triplet = UCShiftTripletLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                           lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1],
                                           lambda_g=cfg.SOLVER.MAG_G,
                                           s=cfg.SOLVER.S, reg_type=cfg.SOLVER.REG_TYPE, detach_pn=True)
        else:
            raise NotImplementedError
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'uc-atri-n':
        if 'uc' in cfg.MODEL.ARCH:
            triplet = UCShiftArcTripletLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                           lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1],
                                           lambda_g=cfg.SOLVER.MAG_G,
                                           s=cfg.SOLVER.S, reg_type=cfg.SOLVER.REG_TYPE)
        else:
            raise NotImplementedError

    else:
        triplet = NoneTri()
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        # raise NotImplementedError

    return triplet


def make_cls_loss(cfg, num_classes, W=None):
    if cfg.MODEL.CLS_LOSS_TYPE == 'cosface':
        cls = CosFaceLoss(m=cfg.SOLVER.MARGIN, s=cfg.SOLVER.CS)
    elif cfg.MODEL.CLS_LOSS_TYPE == 'n-softmax':
        cls = NormFace(s=cfg.SOLVER.CS)
    elif cfg.MODEL.CLS_LOSS_TYPE == 'arcface':
        cls = ArcFaceLoss(m=cfg.SOLVER.MARGIN, s=cfg.SOLVER.CS)
    elif cfg.MODEL.CLS_LOSS_TYPE == 'proj':
        cls = ProjCLSLoss(m=cfg.SOLVER.MARGIN, s=cfg.SOLVER.CS)
    elif 'ucx-cls' in cfg.MODEL.CLS_LOSS_TYPE:
        if 'uc' in cfg.MODEL.ARCH:
            metric = cfg.MODEL.ARCH.split('-')[-1]
            cls = UCShiftCLSLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                 lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                                 s=cfg.SOLVER.CS, reg_type=cfg.SOLVER.REG_TYPE, uc=False, metric=metric)
    # elif cfg.MODEL.CLS_LOSS_TYPE == 'uc-cls':
    elif 'uc-cls' in cfg.MODEL.CLS_LOSS_TYPE:
        if 'uc' in cfg.MODEL.ARCH:
            metric = cfg.MODEL.ARCH.split('-')[-1]
            cls = UCShiftCLSLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                 lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                                 s=cfg.SOLVER.CS, reg_type=cfg.SOLVER.REG_TYPE, metric=metric)
        else:
            cls = UCCLSLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                                 lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                                 s=cfg.SOLVER.CS, reg_type=cfg.SOLVER.REG_TYPE)
    elif cfg.MODEL.CLS_LOSS_TYPE == 'uc-cls-ns':
        cls = UCCLSLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                             lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                             s=cfg.SOLVER.CS, reg_type=cfg.SOLVER.REG_TYPE)
    elif cfg.MODEL.CLS_LOSS_TYPE == 'uc-cls-n':
        cls = UCNormedCLSLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                             lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                             s=cfg.SOLVER.CS, reg_type=cfg.SOLVER.REG_TYPE)
    elif cfg.MODEL.CLS_LOSS_TYPE == 'uc-xent':
        cls = UCXentLoss(margin=cfg.SOLVER.MARGIN, la=cfg.SOLVER.NORM_LU[0], ua=cfg.SOLVER.NORM_LU[1],
                             lm=cfg.SOLVER.MARGIN_LU[0], um=cfg.SOLVER.MARGIN_LU[1], lambda_g=cfg.SOLVER.MAG_G,
                             s=cfg.SOLVER.CS, reg_type=cfg.SOLVER.REG_TYPE)
    elif cfg.MODEL.CLS_LOSS_TYPE == 'ce':
        cls = torch.nn.CrossEntropyLoss()
    elif cfg.MODEL.CLS_LOSS_TYPE == 'softmax':
        cls = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        cls = NoneCLS()
    return cls



def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        # triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        if cfg.SOLVER.TRI_TYPE == 'tri':
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
        elif cfg.SOLVER.TRI_TYPE == 'r-tri':
            triplet = RelativeTripletLoss(cfg.SOLVER.MARGIN)
        elif cfg.SOLVER.TRI_TYPE == 'gr-tri':
            triplet = RelativeTripletLoss(cfg.SOLVER.MARGIN, num_classes, cfg.DATALOADER.NUM_INSTANCE)
        else:
            raise NotImplementedError
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                l_cls =  xent(score, target)
                l_tri = triplet(feat, target)[0]
                l_cent = center_criterion(feat, target)
                return l_cls + cfg.SOLVER.TRI_LOSS_WEIGHT * l_tri + cfg.SOLVER.CENTER_LOSS_WEIGHT * l_cent, l_tri
                # return xent(score, target) + \
                #         triplet(feat, target)[0] + \
                #         cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion


'''
def make_qg_loss(cfg, num_classes):    # modified by gu
    from .qg_triplet_loss import TripletLoss
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cosface':
        triplet = CosfacePairwiseLossqg(cfg.SOLVER.MARGIN, cfg.SOLVER.TRIGAMMA)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=('cuda' in cfg.MODEL.DEVICE))     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    # sim = SimLoss()

    # if sampler == 'softmax':
    #     def loss_func(score, feat, target):
    #         return F.cross_entropy(score, target)
    # elif cfg.DATALOADER.SAMPLER == 'triplet':
    #     def loss_func(score, feat, ftmap, att, target):
    #         return triplet(ftmap, att, target)[0]
    # elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':

        # real loss func
    att_temp = torch.rand(128, 1, 24)        # last dim means numparts
    torch.nn.init.orthogonal_(att_temp)
    att_temp = F.normalize(F.interpolate(att_temp, size=16, mode='linear', align_corners=None), dim=0).squeeze()
    # att_temp = att_temp*5
    def loss_func(sout, tout, target, ratio, start):
        # score, feat, ftmap, att = sout
        l_cls = xent(sout[0], target)
        # l_tri = triplet(sout[2], sout[3], target)[0]

        # fake mask
        b = start.shape[0]
        h = sout[2].shape[-1]
        if cfg.SOLVER.TRAIN_MODE == "base":
            sout = list(sout)
            tout = list(tout)
            sout[3] = att_temp.expand(b, *att_temp.shape).to(target.device)*3
            tout[3] = F.normalize(resize_score(sout[3], 1/ratio, start))*3
        # mask = torch.zeros((b, 1, h), device=start[0].device)
        # for i in range(b):
        #     s = int(start[i]*h)
        #     e = int((start[i]+ratio[i])*h)
        #     mask[i, 0, s:e] = 1/ratio[i]
        # tout[3][:] = (mask * tout[3].fill_(1))[:]
        # sout[3].fill_(1)

        # l_tri = triplet(torch.cat([sout[2],t], dim=0),
        l_tri = triplet(sout[2], sout[3], target)[0] * 0.5
        # l_tri += triplet(tout[2], tout[3], target)[0]
        l_tri += triplet([sout[2],tout[2]], [sout[3],tout[3]], target)[0] * 0.5
        # l_tri += triplet(torch.cat([sout[2],tout[2]], dim=0),
        #                 torch.cat([sout[3],tout[3]], dim=0),
        #                 torch.cat([target,target], dim=0))[0]
        # l_sim, l_dig = sim(sout[3], tout[3], ratio, start)
        # return l_cls, l_tri, l_sim.detach(), l_dig.detach()
        if cfg.DATALOADER.SAMPLER == 'softmax':
            l_tri = l_tri.detach()
        elif cfg.DATALOADER.SAMPLER == 'triplet':
            l_cls = l_cls.detach()
        return l_cls, l_tri
    # else:
    #     print('expected sampler should be softmax, triplet or softmax_triplet, '
    #           'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_stn_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cosface':
        triplet = CosfacePairwiseLossqg(cfg.SOLVER.MARGIN, cfg.SOLVER.TRIGAMMA)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=('cuda' in cfg.MODEL.DEVICE))     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    # real loss func
    def loss_func(sout, tout, target, ratio, start):
        # score, feat, ftmap, att = sout
        l_cls = xent(sout[0], target)
        # l_tri = triplet(sout[2], sout[3], target)[0]

        l_tri = triplet(sout[2], sout[3], target)[0] * 0.5
        # l_tri += triplet(tout[2], tout[3], target)[0]
        l_tri += triplet([sout[2],tout[2]], [sout[3],tout[3]], target)[0] * 0.5
        if cfg.DATALOADER.SAMPLER == 'softmax':
            l_tri = l_tri.detach()
        elif cfg.DATALOADER.SAMPLER == 'triplet':
            l_cls = l_cls.detach()
        return l_cls, l_tri
    return loss_func


def make_attr_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        # triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        if cfg.SOLVER.TRI_TYPE == 'tri':
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
        elif cfg.SOLVER.TRI_TYPE == 'tri-n':
            triplet = TripletLoss(cfg.SOLVER.MARGIN, True)
        elif cfg.SOLVER.TRI_TYPE == 'tri-att':
            from .mutilabel_loss import AttrTripletLoss
            triplet = AttrTripletLoss(cfg.SOLVER.MARGIN, True)
        elif cfg.SOLVER.TRI_TYPE == 'r-tri':
            triplet = RelativeTripletLoss(cfg.SOLVER.MARGIN, gamma=cfg.SOLVER.TRIGAMMA)
        else:
            raise NotImplementedError
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet':
        def loss_func(score, score_att, a_dist, feat, target):
            if not hasattr(loss_func, 'triplet'):
                loss_func.triplet = triplet
                loss_func.weight_t = 1

            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                l_cls = xent(score, target[:, -1])
                l_att = att_loss(score_att, target[:, :-1])
                l_tri = loss_func.triplet(feat, target[:, :], a_dist)[0]   # todo： use attr label
                # l_uc = vis_a.pow(2).mean()     # uncertainty
                return l_cls + l_tri + l_att, l_tri, l_att, a_dist.mean()
                # return l_cls + l_tri + l_att + cfg.SOLVER.ATT_LAMBDA * l_uc, l_tri, l_att, torch.sigmoid(vis_a).mean()
                # return xent(score, target) + triplet(feat, target)[0]
            else:
                return F.cross_entropy(score, target) + triplet(feat, target)[0] + att_loss(score_att, target[:, :-1])
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_region_loss(cfg, num_classes):    # modified by gu
    from .region_loss import RegionLoss
    from .pixel_contrasive_loss import PixelContrasiveLoss
    region = RegionLoss(thr=cfg.SOLVER.REGION_T, npatch=cfg.SOLVER.REGION_N,
        s=cfg.SOLVER.REGION_S, margin=cfg.SOLVER.REGION_M)
    pix_contrasive = PixelContrasiveLoss().to(cfg.MODEL.DEVICE)
    pix_contrasive = torch.nn.DataParallel(pix_contrasive)
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        # triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        if cfg.SOLVER.TRI_TYPE == 'tri':
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
        elif cfg.SOLVER.TRI_TYPE == 'tri-n':
            triplet = TripletLoss(cfg.SOLVER.MARGIN, True)
        elif cfg.SOLVER.TRI_TYPE == 'r-tri':
            triplet = RelativeTripletLoss(cfg.SOLVER.MARGIN, gamma=cfg.SOLVER.TRIGAMMA)
        else:
            raise NotImplementedError
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, ema_feats, proj_feats, p_label, target):
            if not hasattr(loss_func, 'triplet'):
                loss_func.triplet = triplet
                loss_func.weight_t = 1

            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                l_cls = xent(score, target)
                # l_reg, l_regs = region(featmap, target)
                l_con = pix_contrasive(ema_feats, proj_feats, p_label, target).mean()   # pixel wise contrasive
                l_tri = loss_func.triplet(feat, target)[0]
                # l_uc = vis_a.pow(2).mean()     # uncertainty
                return l_cls + l_tri + l_con, l_tri, l_con, l_con
                # return l_cls + l_tri + l_reg + l_regs, l_tri, l_reg, l_regs
                # return l_cls + l_tri + l_att + cfg.SOLVER.ATT_LAMBDA * l_uc, l_tri, l_att, torch.sigmoid(vis_a).mean()
                # return xent(score, target) + triplet(feat, target)[0]
            else:
                return F.cross_entropy(score, target) + triplet(feat, target)[0]
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func

'''