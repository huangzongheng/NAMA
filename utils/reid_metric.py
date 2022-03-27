# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from ignite.metrics import Metric
from tqdm import tqdm
from einops import rearrange
import math
import pdb
import time

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
from .visualize import vis_norm_uc
import logging

# from .visualize import visualize_ranked_results

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', return_AP=False, norm_k=0, use_norm=False):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.dist_mat = None
        self.norms = None
        self.return_AP = return_AP
        # self.use_norm = use_norm
        self.k = norm_k
        self.logger = logging.getLogger("reid_baseline.train")
        self.feats = []
        self.uc = []
        self.pids = []
        self.camids = []
        self.AP = []

    def reset(self):
        self.feats = []
        self.uc = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        # if isinstance(feat, tuple):
        #     self.feats.append(feat[0])
        #     self.uc.append(feat[1])
        # else:
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if feats.shape[-1] > 2048:
            uc = feats[:, -1]
            feats = feats[:, :-1]
        else:
            uc = feats.norm(dim=-1)
        norms = feats.norm(dim=-1)
        self.norms = norms.cpu()
        self.uc = uc.cpu()
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            print("The test feature is normalized")
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_pids = torch.tensor(q_pids)
        q_camids = torch.tensor(q_camids)
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_pids = torch.tensor(g_pids)
        g_camids = torch.tensor(g_camids)
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())

        distmat = torch.cdist(qf,gf) # .numpy()
        qn = norms[:self.num_query]
        if self.k != 0:
            distmat = torch.asin(distmat / 2) * 2
            # gn = norms[self.num_query:]
            # qn = norms[:self.num_query]
            # lm=0.0
            # um=-0.25
            # gm = (um - lm) * gn.clamp(10, 110).sub(10) / 100 + lm
            gm = uc[self.num_query:]
            distmat -= (gm[None, :]*self.k + 0.0) # * mask
        # distmat = distmat.cpu().numpy()

        # distmat = refine_dist(distmat, g_pids, q_pids) #  distmat
        self.dist_mat = distmat
        start = time.time()
        result = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank, self.return_AP)
        self.logger.info('eval time: {:.2f}'.format(time.time()-start))
        # print('eval time: {:.2f}'.format(time.time()-start))
        AP = torch.tensor(result[-1])
        self.AP = AP
        # print('| norm range | avg AP |')
        table = ['', '| norm range | avg AP |', '| ----- | ----- |']
        step = 2
        qmin, qmax = qn.min().item(), qn.max().item()
        step = max(1, int((qmax - qmin)//4))
        for i in range(int(qmin//step) * step, int(qmax//step + 1) * step, step):
            table.append('| {:d}~{:d} | {:.2f} |'.format(i, i+step, 100*AP[(i<qn) & (qn<i+step)].mean().item()))
            # print('| {:d}~{:d} | {:.2f} |'.format(i, i+step, 100*AP[(i<qn) & (qn<i+1)].mean().item()))
        self.logger.info('\n'.join(table))

        # uncertainty
        if len(uc) > 0:
            table = ['', '| uc range | avg AP |', '| ----- | ----- |']
            qn = uc[:self.num_query]
            qmin, qmax = qn.min().item(), qn.max().item()
            step = ((qmax - qmin) / 10)
            for i in range(10):
                i = i*step + qmin
                table.append(
                    '| {:.3f}~{:.3f} | {:.3f} |'.format(i, i + step, 100 * AP[(i < qn) & (qn < i + step)].mean().item()))
                # print('| {:d}~{:d} | {:.2f} |'.format(i, i+step, 100*AP[(i<qn) & (qn<i+1)].mean().item()))
            self.logger.info('\n'.join(table))


        # import pdb
        # pdb.set_trace()
        return result


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.dist_mat = None

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        self.dist_mat = distmat
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

pad = 1
cn = pad*2+1
def get_qg_dist(qfeat, qatt, gfeat, gatt, metric='L2'):
    assert metric in ['L2', 'cos']
    sim_mat = torch.matmul(qatt.transpose(1, 2).unsqueeze(1), gatt.unsqueeze(0)).clamp_max(8)  # bq * bg * nq * ng = nq * c @ c * ng
    # sim_mat = (sim_mat + 1e-4).sqrt()
    qs = torch.softmax(sim_mat, -1) # - 1/qatt.shape[-1]
    gs = torch.softmax(sim_mat, -2) # - 1/qatt.shape[-1]
    qw = F.avg_pool2d(qs, (1, cn), 1, padding=(0, pad)).max(dim=-1)[0]*cn - cn/qatt.shape[-1]    # b*nq
    gw = F.avg_pool2d(gs, (cn, 1), 1, padding=(pad, 0)).max(dim=-2)[0]*cn - cn/qatt.shape[-1]   # b*ng
    # qw = torch.softmax(sim_mat, -2).mean(dim=-1)    # b*nq
    # gw = torch.softmax(sim_mat, -1).mean(dim=-2)    # b*ng
    qf = F.normalize((qw.unsqueeze(2) * qfeat.unsqueeze(1)).mean(-1), dim=-1)         # b*cq
    gf = F.normalize((gw.unsqueeze(2) * gfeat.unsqueeze(0)).mean(-1), dim=-1)
    if metric == 'L2':
        dist = (qf - gf).norm(dim=2)                  # cq * cg
    elif metric == 'cos':
        dist = F.cosine_similarity(qf, gf, dim=2)
    # dist = torch.cdist(qf, gf)
    return dist


class R1_mAP_qg(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_qg, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        # feats = torch.cat(self.feats, dim=0)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print('gathering feature...')
        feats = torch.cat([x[1] for x in self.feats]).to(device)
        att = torch.cat([x[2] for x in self.feats]).to(device)
        # if self.feat_norm == 'yes':
        #     print("The test feature is normalized")
        #     feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        qa = att[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        ga = att[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = np.zeros((m, n))
        torch.cuda.empty_cache()
        for i in tqdm(range(n), desc='calculating dist mat with {}'.format(device)):
            distmat[:, i] = get_qg_dist(qf, qa, gf[i:i+1], ga[i:i+1]).squeeze().cpu().numpy()
            # distmat[i] = get_qg_dist(qf[i:i+1], qa[i:i+1], gf, ga).cpu().numpy()
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print('computing CMC & mAP')
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        # cmc, mAP = [0.99] * 50, 0.9
        del feats, att, qf, qa, gf, ga
        torch.cuda.empty_cache()

        return cmc, mAP

import mat4py
import os
# 利用属性gt重排距离矩阵
def refine_dist(dist_mat, g_pids, q_pids):
    attrs = mat4py.loadmat(os.path.join('./datasets/DukeMTMC-reID','attribute','duke_attribute.mat'))['duke_attribute']
    attr_train = attrs['test']
    cls = set(attr_train.keys())
    cls.discard('image_index')
    cls = list(cls)
    cls.sort()
    c = np.zeros((len(attr_train['image_index']), len(cls)))    # +1 for id label
    for i in range(len(cls)):
        c[:, i] = attr_train[cls[i]]
    train_attr = np.clip(c - 1, 0, 100000).astype(np.int64)
    bit_weight = np.cumprod(np.ones(len(cls), dtype=np.int64)*2)
    att_code = (train_attr * bit_weight).sum(-1).astype(np.int64)           # 二进制序列编码属性
    # mapping = dict(zip([int(i) for i in attr_train['image_index']], range(len(attr_train['image_index']))))
    mapping = dict(zip([int(i) for i in attr_train['image_index']], att_code))      # id -> attribute id
    g_aid = np.array([mapping[i] for i in g_pids])
    q_aid = np.array([mapping[i] for i in q_pids])
    att_incorrect = q_aid[:, None] != g_aid[None, :]

    return dist_mat + 10*att_incorrect

@torch.no_grad()    # n,c,h,w/.. /n,(x,y)/n, topk
def refined_dist_with_focus(qfeatmaps, gfeatmaps, qfocus, qg_index, search_range=4, roi_shape=(3,1)):
    # qind = torch.arange(qfeatmaps.shape[0], device=qfeatmaps.device)
    q_focus_feat = qfeatmaps    # [qind, :, qfocus[:,0], qfocus[:,1]]  # n, c,
    qfocus = qfocus.to(device)
    dist = []
    g_focus = []
    qg_dist_map = torch.zeros(q_focus_feat.shape[0], gfeatmaps.shape[0], *gfeatmaps.shape[-2:])
    # qfs = roi_align(q_focus_feat, qfocus.cumsum(dim=1)[:,:,[1,0]].reshape(-1,1,4).unbind(0), roi_shape, aligned=True)
    for i, (qf, gid) in enumerate(zip(q_focus_feat, qg_index)):
        # get query focus region
        # qf = qf[:, qfocus[i,0,0]:qfocus[i,0,0] + qfocus[i,1,0],
        #      qfocus[i,0,1]:qfocus[i,0,1]+qfocus[i,1,1]].to(device)
        # qf = align_pooling(qf.to(device), qfocus[i])       # 对齐输入框pooling，减少量化误差
        # qf = F.adaptive_avg_pool2d(qf, (1, 1))
        # qf = rearrange(qf, 'c 1 1 -> 1 c 1')
        qf = roi_align(qf[None,].to(device), [qfocus[i].cumsum(dim=0)[:,[1,0]].reshape(1,4)],
                       (qfocus[i, 1, 0].int(), 1), aligned=True).mean(-2, keepdim=True)       # roi align，保持水平方向信息
        qf = qf.flatten()[None, :, None]
        q_channel_weight = 1 / ((qf.squeeze() - qfeatmaps[0].to(device).mean(-1).mean(-1)).abs() + 0.01)[None, :, None]
        qf = F.normalize(qf * q_channel_weight, dim=1)
        # qh_start, qh_end = qfocus[i, :, 0].cumsum(dim=0)//16
        gfs = gfeatmaps[gid, :, (qfocus[i,0,0].int()-search_range).clamp_min(0)
                                :((qfocus[i,0,0]+qfocus[i,1,0]).int()+search_range+1)].to(device)
        # 滑窗寻找最优匹配
        gfs = F.avg_pool2d(gfs, (qfocus[i, 1]).int().tolist(), stride=1)
        # gfs = reshaped_hor_pooling(gfs, (qfocus[i, 1]).int().tolist())      # 水平pooling，竖直reshape
        gh, gw = gfs.shape[-2:]
        gfs = rearrange(gfs, 'n c h w -> n c (h w)')
        gfs = F.normalize(gfs * q_channel_weight, dim=1)
        qg_dist = (qf-gfs).norm(dim=1)
        # pdb.set_trace()

        # 生成距离分布图
        qg_dist_map[i, gid] = qg_dist.max(-1)[0][:, None, None].cpu()        # 填充边界
        start = ((qfocus[i,0,0]-search_range).clamp_min(0) + (qfocus[i,1,0])//2).int(), (qfocus[i,1,1]//2).int()
        # start = (gfeatmaps.shape[-2]-gh)//2, (gfeatmaps.shape[-1]-gw)//2
        qg_dist_map[i, gid, start[0]:start[0]+gh, start[1]:start[1]+gw] = \
            rearrange(qg_dist, 'n (h w) -> n h w', w=gw).cpu()        # 保存距离分布
        d, ind = qg_dist.min(dim=-1)
        g_ind = ind.detach().cpu() + (qfocus[i,0,0].cpu().int()-search_range).clamp_min(0)*gw
        g_focus.append(torch.stack([g_ind//gw, g_ind % gw], dim=-1))
        dist.append(d.detach().cpu())
    dist = torch.stack(dist)
    g_focus = torch.stack(g_focus)
    # g_focus = torch.stack([g_focus//gfeatmaps.shape[-1], g_focus % gfeatmaps.shape[-1]], dim=-1)

    return dist, g_focus, qg_dist_map

def align_pooling(x, bbox):     # ..., h, w; (y0,x0),(y1,x1)
    bbox = bbox.cumsum(dim=0)
    mask_y = torch.zeros(x.shape[-2], device=x.device)
    mask_x = torch.zeros(x.shape[-1], device=x.device)
    for i, mask in enumerate((mask_y, mask_x)):
        mask[math.ceil(bbox[0, i]):math.floor(bbox[1, i])] = 1  # fill interval
        mask[math.floor(bbox[0, i])] = 1 - bbox[0, i]%1     # start
        mask[math.floor(bbox[1, i])] = bbox[1, i] % 1       # end

    mask = mask_y[:, None] * mask_x[None, :]

    return F.adaptive_avg_pool2d(x * mask, (1,1))


def reshaped_hor_pooling(x, out_shape=(3,1)):
    x = F.avg_pool2d(x, (1, out_shape[1]), stride=1)        # 水平方向pooling
    w = torch.eye(out_shape[0], device=x.device)[:, None, :, None].repeat(x.shape[-3], 1, 1, 1)
    x = F.conv2d(x, w, groups=x.shape[-3])
    return x


# class Vis_R1_mAP(Metric):
#     def __init__(self, num_query, max_rank=50, feat_norm='yes', val_loader=None):
#         super(R1_mAP, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm
#         self.val_loader = val_loader
#
#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []
#
#     def update(self, output):
#         feat, pid, camid = output
#         self.feats.append(feat)
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))
#
#     def compute(self):
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm == 'yes':
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)
#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])
#         g_camids = np.asarray(self.camids[self.num_query:])
#         m, n = qf.shape[0], gf.shape[0]
#         distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         distmat.addmm_(1, -2, qf, gf.t())
#         distmat = distmat.cpu().numpy()
#         cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
#
#         # 可视化检索结果
#         if self.val_loader is not None:
#             visualize_ranked_results(
#                 distmat,
#                 self.val_loader,
#                 'image',
#                 width=self.datamanager.width,
#                 height=self.datamanager.height,
#                 save_dir=osp.join(save_dir, 'visrank_'+dataset_name + '_' + title[i]),
#                 topk=visrank_topk
#             )
#
#         return cmc, mAP
