# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
import logging
from einops import rearrange, reduce, repeat


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(dist_mat - is_neg*1000, 1, keepdim=True)
    # dist_ap, relative_p_inds = torch.max(
    #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(dist_mat + is_pos * 1000, 1, keepdim=True)
    # dist_an, relative_n_inds = torch.min(
    #     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, normalize_feature=False):
        self.margin = margin
        self.normalize_feature = normalize_feature
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, weight=None, normalize_feature=False):
        # if normalize_feature:
        if self.normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        # if global_feat.shape[-1] > 2048:        # 前2048维为全局特征，不算triplet loss
        #     local_feat = global_feat[..., 2048:]
        #     global_feat = global_feat[..., :2048]
        # else:
        #     local_feat=None
        if global_feat.dim() > 2:
            dist_mat = euclidean_dist(global_feat[0], global_feat[1])
        else:
            dist_mat = euclidean_dist(global_feat, global_feat)

        # if local_feat is not None:
        #     dist_mat = euclidean_dist(local_feat, local_feat) + dist_mat.detach()
        if weight is not None:      # 压缩正样本类内距离
            mask = labels

        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            # loss = self.ranking_loss(dist_an/(dist_ap.detach() + dist_an), dist_ap/(dist_ap + dist_an.detach()), y)
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class RelativeTripletLoss(nn.Module):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=0.1, normalize_feature=True, num_classes=0, num_instances=4,
                 alpha=0.0, beta=0.9, p=1.0, sigma=100.0, gamma=0):
        super().__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.num_classes = num_classes
        self.nimg = num_instances
        self.centers = None
        self.stds = None
        self.p = p
        self.alpha = alpha
        self.beta = 0.9     # beta
        self.gamma = gamma
        self.count = 0
        self.rdist = 0
        self.sigma = sigma
        self.logger = logging.getLogger("reid_baseline.train")
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()


    def forward(self, global_feat, labels, normalize_feature=False):

        if self.centers is None:
            self.centers = torch.zeros((self.num_classes, *global_feat.shape[1:]), device=global_feat.device)
            self.stds = torch.ones(self.num_classes, device=global_feat.device)
        # update centers
        pids = labels[::self.nimg]
        mean_feat = reduce(global_feat, '(p k) c -> p c', 'mean', k=self.nimg).detach()
        self.centers[pids] += 0.2 * (F.normalize(mean_feat, dim=-1) - self.centers[pids])
        self.centers[pids] = F.normalize(self.centers[pids], dim=1)

        if self.normalize_feature:
            global_feat = F.normalize(global_feat, dim=-1)
        # update stds
        r = (rearrange(global_feat, '(p k) c -> p k c', k=self.nimg).detach() - self.centers[pids][:, None]).norm(dim=-1)
        self.stds[pids] += 0.2 * (r.mean(-1) - self.stds[pids])
        ref_ap = repeat(self.stds[pids], 'p -> (p k)', k=self.nimg)

        # calculate dist
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        # rel_dist = (dist_ap - dist_an) / (torch.max(ref_ap, 0.5*dist_ap.detach()))
        # loss = F.softplus(rel_dist + self.margin, beta=20)
        # grad = torch.sigmoid((rel_dist + self.margin) * 20)
        loss = F.softplus((dist_ap - dist_an) + self.margin * ref_ap, beta=20)
        grad = torch.sigmoid(((dist_ap - dist_an) + self.margin * ref_ap) * 20)

        self.count += 1
        if self.count % 200 == 0:
            self.count = 0
            self.logger.info('ref:{:.3f}, grad:{:.3f}'.format(
                ref_ap.mean().item(), grad.mean().item()
            ))
        # if self.margin is not None:
        #     loss = self.ranking_loss(dist_an/(dg_ap + dist_an) + self.alpha, dist_ap/(dist_ap + dg_an), y)
        # else:
        #     loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss.mean(), dist_ap, dist_an


class MagRelativeTripletLoss(RelativeTripletLoss):

    beta=20

    def __init__(self, normalize_feature=True, num_classes=0, num_instances=4,
                 lm=0.05, um=0.25, lambda_g=35.0, la=10, ua=110):    # 5, 50
        super().__init__(normalize_feature=normalize_feature, num_classes=num_classes, num_instances=num_instances)
        # self.margin = None
        # self.normalize_feature = normalize_feature
        # assert mode in ['all', 'same', 'cross']
        # self.soft_mine = soft_mine
        self.lm = lm
        self.um = um
        self.la = la
        self.ua = ua
        self.lambda_g = lambda_g    # max(lambda_g, ((um-lm )/ (ua-la)) / (1 / la**2 -1 / ua**2))
        self.avg_m = 0
        self.min_m = 0
        self.max_m = 0
        self.avg_l = 0

    @staticmethod
    def get_dist(feat1, feat2):
        return euclidean_dist(feat1, feat2)

    def forward(self, global_feat, labels, weight=None, normalize_feature=False):

        # if self.normalize_feature:
        norms = global_feat.norm(dim=-1)
        self.margin = self.m(norms)
        reg = self.g(norms)

        loss, dist_ap, dist_an = super().forward(global_feat, labels)

        loss = loss + reg - reg.detach()
        self.avg_l += 0.1 * (norms.mean().detach().item() - self.avg_l)
        self.avg_m += 0.1 * (self.margin.mean().detach().item() - self.avg_m)
        self.min_m += 0.1 * (self.margin.min().detach().item() - self.min_m)
        self.max_m += 0.1 * (self.margin.max().detach().item() - self.max_m)

        return loss, dist_ap, dist_an

    def m(self, norms):
        # grad: um-lm / ua-la
        norms = norms.clamp(self.la, self.ua)
        x = (norms - self.la) / (self.ua - self.la)
        margin = (self.um - self.lm) * x + self.lm
        return margin

    def g(self, norms):
        # min: norm = ua
        # grad: 1 / ua^2 -1 / norm^2
        # lambda_g > (um-lm / ua-la) / (1 / ua^2 -1 / la^2)
        norms = norms.clamp(self.la, self.ua)
        normed_x = ((norms - self.ua) / (self.la - self.ua))    # la:1, ua:0
        # reg = 1 / norms + norms / self.ua ** 2        # magface
        # reg = normed_x ** 2    # square
        reg = torch.exp(normed_x) - normed_x    # exp
        reg = reg.mean()
        return (reg) * self.lambda_g


# class TripletPosLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None, normalize_feature=False):
#         self.margin = margin
#         self.normalize_feature = normalize_feature
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         # if normalize_feature:
#         if self.normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         if global_feat.dim() > 2:
#             dist_mat = euclidean_dist(global_feat[0], global_feat[1])
#         else:
#             dist_mat = euclidean_dist(global_feat, global_feat)
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         dist_an = dist_an.detach()
#
#         dist_ap, dist_an = (dist_ap/(dist_ap + dist_an.detach())), (dist_an/(dist_ap.detach() + dist_an))
#         # dist_an = (dist_an/(dist_ap + dist_an))
#         if self.margin is not None:
#             # loss = self.ranking_loss(dist_an/(dist_ap.detach() + dist_an), dist_ap/(dist_ap + dist_an.detach()), y)
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an


def soft_hard_example_mining(dist_mat, labels, gamma=32.0):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap = torch.logsumexp(
        gamma * dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)/gamma
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an = -torch.logsumexp(
        -gamma * dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)/gamma
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)


    return dist_ap, dist_an

# bug 点高的版本，detach an
# class RelativeTripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None, num_classes=0, num_instances=4, alpha=0.0, beta=0.9, p=1.0, sigma=100.0, gamma=0):
#         self.margin = margin
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.num_instances = num_instances
#         self.dist_bank = None     # 记录每个class的平均类内距离和类间距离
#         self.p = p
#         self.count = 0
#         self.sigma = sigma
#         self.logger = logging.getLogger("reid_baseline.train")
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if self.dist_bank is None:
#             self.dist_bank = torch.ones(self.num_classes + 1, 4, device=global_feat.device)
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dg_ap = None    # referance distance
#         dg_an = None
#         # calculate average inner and inter class distances
#         if self.num_classes > 0:
#             mask = labels.expand(*labels.shape, *labels.shape) == labels.expand(*labels.shape, *labels.shape).t()  # mask = mask.float()
#             # d_inner = mask*dist_mat.detach()
#             # d_inter = (mask.logical_not())*dist_mat.detach()
#             # d_inner = dist_mat[mask].detach().reshape(mask.shape[0], -1)
#             org_dist = dist_mat.detach()
#             d_inner = dist_mat[mask & ~torch.eye(mask.shape[0], dtype=torch.bool, device=labels.device)]\
#                 .detach().reshape(mask.shape[0], -1)
#             d_inter = dist_mat[torch.logical_not(mask)].detach().reshape(mask.shape[0], -1)
#             d_inner = d_inner.sort(-1, True)[0]     # [:, :math.ceil(self.p * d_inner.shape[-1])]
#             d_inter = d_inter.sort(-1, False)[0][:, :math.ceil(self.p * d_inter.shape[-1])]
#             # d_inter = (1 - mask)*dist_mat.detach()
#             std_inner = d_inner.std()       # -1
#             std_inter = d_inter.std()
#             # print(d_inner.max(), d_inner.mean(), d_inner.min())
#             # if d_inner.max() > 3*d_inner.mean():
#             #     print(d_inner)
#             d_inner = d_inner.mean(-1)
#             d_inter = d_inter.mean(-1)
#             # dg_ap = self.alpha * d_inner + (1 - self.alpha) * self.dist_bank[labels, 0]
#             # dg_an = self.alpha * d_inter + (1 - self.alpha) * self.dist_bank[labels, 1]
#             # 计算全局参考距离均值方差
#             mean_ap = self.alpha * d_inner.mean() + (1 - self.alpha) * self.dist_bank[-1, 0]
#             mean_an = self.alpha * d_inter.mean() + (1 - self.alpha) * self.dist_bank[-1, 1]
#             stdg_ap = self.alpha * std_inner + (1 - self.alpha) * self.dist_bank[-1, 2]     # labels
#             stdg_an = self.alpha * std_inter + (1 - self.alpha) * self.dist_bank[-1, 3]
#             # dg_ap = self.alpha * dist_ap.detach() + (1 - self.alpha) * self.dist_bank[labels, 0]
#             # dg_an = self.alpha * dist_an.detach() + (1 - self.alpha) * self.dist_bank[labels, 1]
#             clabel = labels[::self.num_instances]   # label of each class
#
#             d_inner = d_inner.reshape(-1,self.num_instances)
#             d_inter = d_inter.reshape(-1,self.num_instances)
#             # std_inner = std_inner.reshape(-1,self.num_instances)
#             # std_inter = std_inter.reshape(-1,self.num_instances)
#             # new_dist = torch.stack([d_inner.detach(), d_inter.detach(),
#             #                         std_inner.detach(), std_inter.detach()], dim=-1).mean(1)     # new average dist
#             new_dist = torch.stack([d_inner.detach(), d_inter.detach()], dim=-1).mean(1)     # new average dist
#             # new_dist = torch.stack([dist_ap.detach(), dist_an.detach()], dim=-1)\
#             #     .reshape(-1, self.num_instances, 2).mean(1)     # new average dist
#
#             # 更新参考距离
#             self.dist_bank[clabel, :2] = self.beta * self.dist_bank[clabel, :2] + (1 - self.beta) * new_dist
#             # 更新全局均值与方差
#             if self.dist_bank[-1].std() < 1e-6:
#                 self.dist_bank[-1, :] = torch.stack([d_inner.mean().detach(), d_inter.mean().detach(),
#                                                      std_inner.detach(), std_inter.detach()])
#                 # mean_ap, mean_an = d_inner.mean().detach(), d_inter.mean().detach()
#                 # self.dist_bank[-1, :] = self.beta * self.dist_bank[-1, :] + (1 - self.beta) * torch.stack(
#                 #     [mean_ap.detach(), mean_an.detach(), std_inner.detach(), std_inter.detach()])
#                 # self.dist_bank[-1, :] = torch.tensor([15,15,1,1.], device=self.dist_bank.device)
#                 self.logger.info('initializing dist bank {}'.format(self.dist_bank[-1, :]))
#             else:
#                 self.dist_bank[-1, :] = self.beta * self.dist_bank[-1, :] + (1 - self.beta) * torch.stack(
#                     [mean_ap.detach(), mean_an.detach(), std_inner.detach(), std_inter.detach()])
#                     # [d_inner.mean().detach(), d_inter.mean().detach(), std_inner.detach(), std_inter.detach()])
#
#             # very hard triplets filter
#             if self.sigma < 99:
#                 vhard_ap = (dist_mat > (mean_ap + self.sigma * stdg_ap)) & mask   # d_inner.flatten()
#                 dist_mat = dist_mat * (1 - vhard_ap.float())
#                 vhard_an = (dist_mat < (mean_an - (self.sigma) * stdg_an)) & (~mask)
#                 dist_mat = dist_mat + 100 * vhard_an.float()    # * (1 + 10 * vhard_an.float())
#                 # if self.sigma > 0:
#                 #     vhard_ap = (dist_mat > (self.sigma * d_inner.mean().detach())) & mask   # d_inner.flatten()
#                 #     # vhard_ap = (dist_mat > (mean_ap + self.sigma * stdg_ap)) & mask   # d_inner.flatten()
#                 #     dist_mat = dist_mat * (1 - vhard_ap.float())
#                 # else:
#                 #     vhard_an = (dist_mat < (mean_an - (-self.sigma) * stdg_an)) & (~mask)
#                 #     dist_mat = dist_mat + 100 * vhard_an.float()    # * (1 + 10 * vhard_an.float())
#                 # if vhard_ap.sum() + vhard_an.sum() > 0:
#                 #     self.logger.info('Hard example:p{} {} \nn{} {}\ndist bank {}'.format(
#                 #         vhard_ap.sum(), org_dist[vhard_ap], vhard_an.sum(), org_dist[vhard_an],
#                 #         self.dist_bank[-1, :]))
#                 # if vhard_ap.sum() > 0:
#                 #     # vhard_ap = vhard_ap
#                 #     self.logger.info('hp:{:.3f} {:.3f}/{:.3f}'.format(
#                 #         vhard_ap.sum(), org_dist[vhard_ap].min(), org_dist[vhard_ap].max()))
#                 #     # print('hp:', vhard_ap.sum(), org_dist[vhard_ap])
#                 #     pass
#                 # if vhard_an.sum() > 0:
#                 #     # vhard_an = vhard_an
#                 #     self.logger.info('hn:{:.3f} {:.3f}/{:.3f}'.format(
#                 #         vhard_an.sum(), org_dist[vhard_an].min(), org_dist[vhard_an].max()))
#                 #     # print('hn:', vhard_an.sum(), org_dist[vhard_an])
#                 #     pass
#             else:
#                 # vhard_an = (dist_mat < (mean_an - (self.sigma) * stdg_an)) & (~mask)
#                 dist_mat = dist_mat + 0 * (~mask).float()
#             dg_ap = self.dist_bank[labels, 0].detach()
#             dg_an = self.dist_bank[labels, 1].detach()
#
#
#         else:
#             # dg_ap = dist_ap.detach()    # referance distance
#             # dg_an = dist_an.detach()
#             pass
#
#
#         # batch hard triplet sample
#         if self.gamma > 1e-3:
#             dist_ap, dist_an = soft_hard_example_mining(
#                 dist_mat, labels)
#         else:
#             dist_ap, dist_an = hard_example_mining(
#                 dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#
#         if dg_an is None:
#             dg_ap = dist_ap.detach()    # referance distance
#             dg_an = dist_an.detach()
#         # mn = (dist_an/(dg_ap + dist_an)).detach()
#         mp = (dist_ap/(dist_ap + dg_an)).detach()
#         mn = (dg_an/(dg_ap + dg_an)).detach()
#         # mp = (dg_ap/(dg_ap + dg_an)).detach()
#         self.count += 1
#         if self.count >= 20:
#             self.count = 0
#             r_distp = d_inner.mean() / global_feat.norm(dim=-1).mean()
#             r_distn = d_inter.mean() / global_feat.norm(dim=-1).mean()
#             self.logger.info('mp:{:.3f}({:.3f}/{:.3f}) mn:{:.3f}({:.3f}/{:.3f}) rd:{:.3f}/{:.3f} eft:{}'.format(
#                 mp.mean(), mp.min(), mp.max(), mn.mean(), mn.min(), mn.max(), r_distp, r_distn,
#                 (1 - dist_ap/(dist_ap + dg_an) < self.margin).sum().item()))
#         dist_an = dist_an.detach()
#         # dist_ap = dist_ap.detach()
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an/(dg_ap.zero_() + dist_an), dist_ap/(dist_ap + dg_an), y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an

# class RelativeTripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None, num_classes=0, num_instances=4, alpha=0.0, beta=0.9, p=1.0, sigma=100.0, gamma=0):
#         self.margin = margin
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = 0.9     # beta
#         self.gamma = gamma
#         self.num_instances = num_instances
#         self.dist_bank = None     # 记录每个class的平均类内距离和类间距离
#         self.p = p
#         self.count = 0
#         self.rdist = 0
#         self.sigma = sigma
#         self.logger = logging.getLogger("reid_baseline.train")
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if self.dist_bank is None:
#             self.dist_bank = torch.ones(self.num_classes + 1, 4, device=global_feat.device)
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dg_ap = None    # referance distance
#         dg_an = None
#         # calculate average inner and inter class distances
#         if self.num_classes > 0:
#             mask = labels.expand(*labels.shape, *labels.shape) == labels.expand(*labels.shape, *labels.shape).t()  # mask = mask.float()
#             org_dist = dist_mat.detach()
#             d_inner = dist_mat[mask & ~torch.eye(mask.shape[0], dtype=torch.bool, device=labels.device)] \
#                 .detach().reshape(mask.shape[0], -1)
#             d_inter = dist_mat[torch.logical_not(mask)].detach().reshape(mask.shape[0], -1)
#             d_inner = d_inner.sort(-1, True)[0]     # [:, :math.ceil(self.p * d_inner.shape[-1])]
#             d_inter = d_inter.sort(-1, False)[0][:, :math.ceil(self.p * d_inter.shape[-1])]
#             # d_inter = (1 - mask)*dist_mat.detach()
#             std_inner = d_inner.std()       # -1
#             std_inter = d_inter.std()
#             # print(d_inner.max(), d_inner.mean(), d_inner.min())
#             # if d_inner.max() > 3*d_inner.mean():
#             #     print(d_inner)
#             d_inner = d_inner.mean(-1)
#             d_inter = d_inter.mean(-1)
#             # dg_ap = self.alpha * d_inner + (1 - self.alpha) * self.dist_bank[labels, 0]
#             # dg_an = self.alpha * d_inter + (1 - self.alpha) * self.dist_bank[labels, 1]
#             # 计算全局参考距离均值方差
#             mean_ap = self.alpha * d_inner.mean() + (1 - self.alpha) * self.dist_bank[-1, 0]
#             mean_an = self.alpha * d_inter.mean() + (1 - self.alpha) * self.dist_bank[-1, 1]
#             stdg_ap = self.alpha * std_inner + (1 - self.alpha) * self.dist_bank[-1, 2]     # labels
#             stdg_an = self.alpha * std_inter + (1 - self.alpha) * self.dist_bank[-1, 3]
#             # dg_ap = self.alpha * dist_ap.detach() + (1 - self.alpha) * self.dist_bank[labels, 0]
#             # dg_an = self.alpha * dist_an.detach() + (1 - self.alpha) * self.dist_bank[labels, 1]
#             clabel = labels[::self.num_instances]   # label of each class
#
#             d_inner = d_inner.reshape(-1,self.num_instances)
#             d_inter = d_inter.reshape(-1,self.num_instances)
#             new_dist = torch.stack([d_inner.detach(), d_inter.detach()], dim=-1).mean(1)     # new average dist
#             # new_dist = torch.stack([dist_ap.detach(), dist_an.detach()], dim=-1)\
#             #     .reshape(-1, self.num_instances, 2).mean(1)     # new average dist
#
#             # 更新参考距离
#             self.dist_bank[clabel, :2] = self.beta * self.dist_bank[clabel, :2] + (1 - self.beta) * new_dist
#             # 更新全局均值与方差
#             if self.dist_bank[-1].std() < 1e-6:
#                 self.dist_bank[-1, :] = torch.stack([d_inner.mean().detach(), d_inter.mean().detach(),
#                                                      std_inner.detach(), std_inter.detach()])
#                 self.logger.info('initializing dist bank {}'.format(self.dist_bank[-1, :]))
#             else:
#                 self.dist_bank[-1, :] = self.beta * self.dist_bank[-1, :] + (1 - self.beta) * torch.stack(
#                     [mean_ap.detach(), mean_an.detach(), std_inner.detach(), std_inter.detach()])
#                 # [d_inner.mean().detach(), d_inter.mean().detach(), std_inner.detach(), std_inter.detach()])
#
#             # very hard triplets filter
#             dg_ap = self.dist_bank[labels, 0].detach()
#             dg_an = self.dist_bank[labels, 1].detach()
#
#
#         else:
#             # dg_ap = dist_ap.detach()    # referance distance
#             # dg_an = dist_an.detach()
#             pass
#
#
#         # batch hard triplet sample
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#
#         if dg_an is None:
#             dg_ap = dist_ap.detach()    # referance distance
#             dg_an = dist_an.detach()
#         # mn = (dist_an/(dg_ap + dist_an)).detach()
#         mp = (dist_ap/(dist_ap + dg_an)).detach()
#         mn = (dg_an/(dg_ap + dg_an)).detach()
#         # mp = (dg_ap/(dg_ap + dg_an)).detach()
#         self.count += 1
#         if self.count >= 20:
#             self.count = 0
#             r_distp = d_inner.mean() / global_feat.norm(dim=-1).mean()
#             r_distn = d_inter.mean() / global_feat.norm(dim=-1).mean()
#             self.logger.info('rd:{:.3f}/{:.3f} eft:{}'.format(
#                 r_distp, r_distn,
#                 (1 - dist_ap/(dist_ap + dg_an) + self.alpha < self.margin).sum().item()))
#         # dist_an = dist_an.detach()
#         # dist_ap = dist_ap.detach()
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an/(dg_ap + dist_an) + self.alpha, dist_ap/(dist_ap + dg_an), y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an
#
#
# class BaseRelativeTripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None, num_classes=0, num_instances=4, alpha=0.0, beta=0.9, p=1.0, sigma=100.0, gamma=0):
#         self.margin = margin
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.ad_margin = [0, 0]
#         self.num_instances = num_instances
#         self.dist_bank = None     # 记录每个class的平均类内距离和类间距离
#         self.avg_rdist = None
#         self.p = p
#         self.sigma = sigma
#         self.softplus = torch.nn.Softplus(50)
#         self.logger = logging.getLogger("reid_baseline.train")
#         self.count = 0
#         self.rdist = 0
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=min(margin, 1))
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if self.avg_rdist is None:
#             self.avg_rdist = torch.ones(3, 2, device=global_feat.device)
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dg_ap = None    # referance distance
#         dg_an = None
#         # calculate average inner and inter class distances
#         # if self.num_classes > 0:
#         mask = labels.expand(*labels.shape, *labels.shape) == labels.expand(*labels.shape, *labels.shape).t()  # mask = mask.float()
#         # d_inner = mask*dist_mat.detach()
#         # d_inter = (mask.logical_not())*dist_mat.detach()
#         # d_inner = dist_mat[mask].detach().reshape(mask.shape[0], -1)
#         org_dist = dist_mat.detach()
#         d_inner = dist_mat[mask & ~torch.eye(mask.shape[0], dtype=torch.bool, device=labels.device)] \
#             .detach().reshape(mask.shape[0], -1)
#         d_inter = dist_mat[torch.logical_not(mask)].detach().reshape(mask.shape[0], -1)
#         # d_inner = d_inner * ((d_inner < 3 * d_inner.mean()).float())
#         # d_inter = d_inter * ((d_inter * 3 > d_inner.mean()).float())
#
#         d_inner = d_inner.sort(-1, True)[0]     # [:, :math.ceil(self.p * d_inner.shape[-1])]
#         d_inter = d_inter.sort(-1, False)[0]        # [:, :math.ceil(self.p * d_inter.shape[-1])]
#         # d_inter = (1 - mask)*dist_mat.detach()
#         std_inner = d_inner.std()       # -1
#         std_inter = d_inter.std()
#         # print(d_inner.max(), d_inner.mean(), d_inner.min())
#         # if d_inner.max() > 3*d_inner.mean():
#         #     print(d_inner)
#         dg_ap = d_inner.mean(-1).detach()   # 平均参考距离
#         dg_an = d_inter.mean(-1).detach()
#         # dg_ap = d_inner[..., 0].detach()   # 最大值作为参考距离
#         # dg_an = d_inter[..., 0].detach()
#         r_distp = d_inner.mean() / global_feat.norm(dim=-1).mean()
#         r_distn = d_inter.mean() / global_feat.norm(dim=-1).mean()
#         # if self.avg_rdist.std() < 1e-6:
#         #     self.avg_rdist[:] = torch.stack([r_distp, r_distn]).detach()
#         # else:
#         #     self.avg_rdist[0] = torch.stack([r_distp, r_distn]).detach()
#         #     self.avg_rdist[1] = self.beta * self.avg_rdist[1] + (1 - self.beta) * self.avg_rdist[0]
#         #     self.avg_rdist[2] = self.beta * self.avg_rdist[2] + (1 - self.beta) * self.avg_rdist[1]
#
#         # batch hard triplet sample
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         # dist_an = dist_an.detach()
#
#         if dg_an is None:
#             dg_ap = dist_ap.detach()    # referance distance
#             dg_an = dist_an.detach()
#         mn = (dist_an/(dg_ap + dist_an)).detach()
#         mp = (dist_ap/(dist_ap + dg_an)).detach()
#         self.count += 1
#         if self.count >= 20:
#             self.count = 0
#             # self.logger.info('mp:{:.3f}({:.3f}/{:.3f}) mn:{:.3f}({:.3f}/{:.3f}) rdist:{:.3f}/{:.3f}'.format(
#             #     mp.mean(), mp.min(), mp.max(), mn.mean(), mn.min(), mn.max(), r_distp, r_distn))
#             self.logger.info('rd:{:.3f}/{:.3f} eft:{}'.format(
#                 r_distp, r_distn, (dist_an/(dg_ap + dist_an) + self.alpha - dist_ap/(dist_ap + dg_an) < self.margin).sum().item()))
#             # self.avg_rdist.flatten(), self.avg_rdist[2] - self.avg_rdist[1]))
#         self.rdist = (dist_ap/(dist_ap + dg_an)).detach()
#         if self.margin == 0:
#             # if self.margin >= 0.9999:
#             # mp = (dist_an/(dg_ap + dist_an) - 0.5).detach()
#             # mn = (0.5 - dist_ap/(dist_ap + dg_an)).detach()
#
#             # loss = (self.ranking_loss(dist_an/(dg_ap + dist_an), 0.5 - mn.mean().expand(*dist_an.shape), y)
#             #         + self.ranking_loss(mp.mean().expand(*dist_an.shape) + 0.5, dist_ap/(dist_ap + dg_an), y))
#             loss = self.softplus(mn.mean().clamp(0.5 + self.alpha, 1) - dist_an/(dg_ap + dist_an)) \
#                    + self.softplus(dist_ap/(dist_ap + dg_an) - mp.mean().clamp(0, .5 - self.alpha))
#             loss = 2 * loss.mean()
#             # loss = (self.ranking_loss(dist_an/(dg_ap + dist_an), mn.mean().expand(*dist_an.shape), y)
#             #         + self.ranking_loss(mp.mean().expand(*dist_an.shape), dist_ap/(dist_ap + dg_an), y))
#             # loss = (self.ranking_loss(2 * dist_an/(dg_ap + dist_an), 2*mn.mean() + torch.ones_like(dist_an, device=dist_an.device), y)
#             #         + self.ranking_loss(2*mp.mean() + torch.ones_like(dist_an, device=dist_an.device), 2 * dist_ap/(dist_ap + dg_an), y)) / 2
#         elif self.margin is not None:
#             loss = (self.ranking_loss(dist_an/(dg_ap + dist_an) + self.alpha, dist_ap/(dist_ap + dg_an), y))
#             # loss = (self.ranking_loss(2 * dist_an/(dg_ap + dist_an), torch.ones_like(dist_an, device=dist_an.device), y)
#             #         + self.ranking_loss(torch.ones_like(dist_an, device=dist_an.device), 2 * dist_ap/(dist_ap + dg_an), y)) / 2
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an
#
#
# class TightRelativeTripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     # 主要通过约束类内距离来使得每个类更紧凑，类间距离仅约束类间平均距离，不针对单个样本
#     def __init__(self, margin=None, num_classes=0, num_instances=4, alpha=0.0, beta=0.9, p=1.0, sigma=100.0, gamma=0):
#         self.margin = margin
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.ad_margin = [0, 0]
#         self.num_instances = num_instances
#         self.dist_bank = None     # 记录每个class的平均类内距离和类间距离
#         self.avg_rdist = None
#         self.p = p
#         self.sigma = sigma
#         self.softplus = torch.nn.Softplus(50)
#         self.logger = logging.getLogger("reid_baseline.train")
#         self.count = 0
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=min(margin, 1))
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if self.avg_rdist is None:
#             self.avg_rdist = torch.ones(3, 2, device=global_feat.device)
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         class_feat = global_feat.reshape(-1, self.num_instances, global_feat.shape[-1]).mean(1)
#         inter_dist_mat = euclidean_dist(class_feat , class_feat)
#         dg_ap = None    # referance distance
#         dg_an = None
#         # calculate average inner and inter class distances
#         # if self.num_classes > 0:
#         mask = labels.expand(*labels.shape, *labels.shape) == labels.expand(*labels.shape, *labels.shape).t()  # mask = mask.float()
#         # d_inner = mask*dist_mat.detach()
#         # d_inter = (mask.logical_not())*dist_mat.detach()
#         # d_inner = dist_mat[mask].detach().reshape(mask.shape[0], -1)
#         org_dist = dist_mat.detach()
#         d_inner = dist_mat[mask & ~torch.eye(mask.shape[0], dtype=torch.bool, device=labels.device)] \
#             .detach().reshape(mask.shape[0], -1)
#         d_inter = inter_dist_mat.detach()
#         # d_inter = d_inter.reshape(d_inter.shape[0]//self.num_instances, self.num_instances, -1, self.num_instances).mean((1,3))
#         # d_inter = dist_mat[torch.logical_not(mask)].detach().reshape(mask.shape[0], -1)
#         # d_inner = d_inner * ((d_inner < 3 * d_inner.mean()).float())
#         # d_inter = d_inter * ((d_inter * 3 > d_inner.mean()).float())
#
#         d_inner = d_inner.sort(-1, True)[0]     # [:, :math.ceil(self.p * d_inner.shape[-1])]
#         d_inter = d_inter.sort(-1, False)[0][:, 1:]        # [:, :math.ceil(self.p * d_inter.shape[-1])]
#
#         std_inner = d_inner.std()       # -1
#         std_inter = d_inter.std()
#         # print(d_inner.max(), d_inner.mean(), d_inner.min())
#         # if d_inner.max() > 3*d_inner.mean():
#         #     print(d_inner)
#         # dg_ap = d_inner.mean(-1).detach()   # 平均参考距离
#         # dg_an = d_inter.mean(-1).detach()
#         dg_ap = d_inner[..., 0].detach()   # 最大值作为参考距离
#         dg_an = d_inter[..., 0].detach()
#         dg_ap = dg_ap.reshape(self.num_instances, *dg_an.shape).mean(0)
#         dg_an = dg_an.expand(self.num_instances, *dg_an.shape).flatten()
#         r_distp = d_inner.mean() / global_feat.norm(dim=-1).mean()
#         r_distn = d_inter.mean() / global_feat.norm(dim=-1).mean()
#         if self.avg_rdist.std() < 1e-6:
#             self.avg_rdist[:] = torch.stack([r_distp, r_distn]).detach()
#         else:
#             self.avg_rdist[0] = torch.stack([r_distp, r_distn]).detach()
#             self.avg_rdist[1] = self.beta * self.avg_rdist[1] + (1 - self.beta) * self.avg_rdist[0]
#             self.avg_rdist[2] = self.beta * self.avg_rdist[2] + (1 - self.beta) * self.avg_rdist[1]
#
#         # batch hard triplet sample
#         dist_ap, _ = hard_example_mining(
#             dist_mat, labels)
#         _, dist_an = hard_example_mining(
#             inter_dist_mat, labels[::self.num_instances])
#         yp = dist_ap.new().resize_as_(dist_ap).fill_(1)
#         yn = dist_an.new().resize_as_(dist_an).fill_(1)
#
#         if dg_an is None:
#             dg_ap = dist_ap.detach()    # referance distance
#             dg_an = dist_an.detach()
#         mn = (dist_an/(dg_ap + dist_an)).detach()
#         mp = (dist_ap/(dist_ap + dg_an)).detach()
#         self.count += 1
#         if self.count >= 20:
#             self.count = 0
#             self.logger.info('mp:{:.3f}({:.3f}/{:.3f}) mn:{:.3f}({:.3f}/{:.3f}) rd:{:.3f}/{:.3f}'.format(
#                 mp.mean(), mp.min(), mp.max(), mn.mean(), mn.min(), mn.max(), r_distp, r_distn))
#         loss = (self.ranking_loss(torch.ones_like(dist_ap, device=dist_ap.device), 2*dist_ap/(dist_ap + dg_an), yp)
#                 + self.ranking_loss(2*dist_an/(dg_ap + dist_an), torch.ones_like(dist_an, device=dist_an.device), yn)
#                 # + self.ranking_loss(dist_an, torch.ones_like(dist_an, device=dist_an.device), yn)
#                 )/2
#         return loss, dist_ap, dist_an
#
# # 优化最大类内距离和最小类间距离组成的triplet
# class ClassRelativeTripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None, num_classes=0, num_instances=4, alpha=0.0, beta=0.9, p=1.0, sigma=100.0, gamma=0):
#         self.margin = margin
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.ad_margin = [0, 0]
#         self.num_instances = num_instances
#         self.dist_bank = None     # 记录每个class的平均类内距离和类间距离
#         self.avg_rdist = None
#         self.p = p
#         self.sigma = sigma
#         self.softplus = torch.nn.Softplus(50)
#         self.logger = logging.getLogger("reid_baseline.train")
#         self.count = 0
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=min(margin, 1))
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if self.avg_rdist is None:
#             self.avg_rdist = torch.ones(3, 2, device=global_feat.device)
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         class_feat = global_feat.reshape(-1, self.num_instances, global_feat.shape[-1]).mean(1)
#         inter_dist_mat = euclidean_dist(class_feat , class_feat)
#         dg_ap = None    # referance distance
#         dg_an = None
#         # calculate average inner and inter class distances
#         # if self.num_classes > 0:
#         mask = labels.expand(*labels.shape, *labels.shape) == labels.expand(*labels.shape, *labels.shape).t()  # mask = mask.float()
#         org_dist = dist_mat.detach()
#         d_inner = dist_mat[mask & ~torch.eye(mask.shape[0], dtype=torch.bool, device=labels.device)] \
#             .detach().reshape(mask.shape[0], -1)
#         d_inter = inter_dist_mat.detach()
#
#         d_inner = d_inner.sort(-1, True)[0]     # [:, :math.ceil(self.p * d_inner.shape[-1])]
#         d_inter = d_inter.sort(-1, False)[0][:, 1:]        # [:, :math.ceil(self.p * d_inter.shape[-1])]
#
#         std_inner = d_inner.std()       # -1
#         std_inter = d_inter.std()
#         # dg_ap = d_inner[..., 0].detach()   # 最大值作为参考距离
#         # dg_an = d_inter[..., 0].detach()
#         # dg_ap = dg_ap.reshape(self.num_instances, *dg_an.shape).mean(0)
#         # dg_an = dg_an.expand(self.num_instances, *dg_an.shape).flatten()
#         r_distp = d_inner.mean() / global_feat.norm(dim=-1).mean()
#         r_distn = d_inter.mean() / global_feat.norm(dim=-1).mean()
#
#         # batch hard triplet sample
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         dist_ap = dist_ap.reshape(-1, self.num_instances).max(dim=-1)[0]
#         dist_an = dist_an.reshape(-1, self.num_instances).min(dim=-1)[0]
#         # dist_ap = self.softplus(dist_mat[mask].reshape(-1, self.num_instances * self.num_instances))    # [C]
#
#         # dist_ap, _ = hard_example_mining(
#         #     dist_mat, labels)
#         # _, dist_an = hard_example_mining(
#         #     inter_dist_mat, labels[::self.num_instances])
#         y = dist_ap.new().resize_as_(dist_ap).fill_(1)
#         # yn = dist_an.new().resize_as_(dist_an).fill_(1)
#
#         if dg_an is None:
#             dg_ap = dist_ap.detach()    # referance distance
#             dg_an = dist_an.detach()
#         mn = (dist_an/(dg_ap + dist_an)).detach()
#         mp = (dist_ap/(dist_ap + dg_an)).detach()
#         self.count += 1
#         if self.count >= 20:
#             self.count = 0
#             self.logger.info('mp:{:.3f}({:.3f}/{:.3f}) mn:{:.3f}({:.3f}/{:.3f}) rd:{:.3f}/{:.3f}'.format(
#                 mp.mean(), mp.min(), mp.max(), mn.mean(), mn.min(), mn.max(), r_distp, r_distn))
#         loss = (self.ranking_loss(2 * dist_an/(dg_ap + dist_an), torch.ones_like(dist_an, device=dist_an.device), y)
#                 + self.ranking_loss(torch.ones_like(dist_an, device=dist_an.device), 2 * dist_ap/(dist_ap + dg_an), y)
#                 ) / 2
#
#         return loss, dist_ap, dist_an
#
# # 用focal loss思想实现adaptive margin
# class FocalRelativeTripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None, num_classes=0, num_instances=4, alpha=0.0, beta=0.9, p=1.0, sigma=100.0, gamma=0):
#         self.margin = margin
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.ad_margin = [0, 0]
#         self.num_instances = num_instances
#         self.dist_bank = None     # 记录每个class的平均类内距离和类间距离
#         self.avg_rdist = None
#         self.p = p
#         self.sigma = sigma
#         self.softplus = torch.nn.Softplus(50)
#         self.logger = logging.getLogger("reid_baseline.train")
#         self.count = 0
#         self.rdist = 0
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=min(margin, 1))
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if self.dist_bank is None:
#             self.dist_bank = torch.ones(self.num_classes + 1, 4, device=global_feat.device)
#         if self.avg_rdist is None:
#             self.avg_rdist = torch.ones(3, 2, device=global_feat.device)
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dg_ap = None    # referance distance
#         dg_an = None
#
#         mask = labels.expand(*labels.shape, *labels.shape) == labels.expand(*labels.shape, *labels.shape).t()  # mask = mask.float()
#         # d_inner = mask*dist_mat.detach()
#         # d_inter = (mask.logical_not())*dist_mat.detach()
#         # d_inner = dist_mat[mask].detach().reshape(mask.shape[0], -1)
#         org_dist = dist_mat.detach()
#         d_inner = dist_mat[mask & ~torch.eye(mask.shape[0], dtype=torch.bool, device=labels.device)] \
#             .detach().reshape(mask.shape[0], -1)
#         d_inter = dist_mat[torch.logical_not(mask)].detach().reshape(mask.shape[0], -1)
#
#         d_inner = d_inner.sort(-1, True)[0]     # [:, :math.ceil(self.p * d_inner.shape[-1])]
#         d_inter = d_inter.sort(-1, False)[0]        # [:, :math.ceil(self.p * d_inter.shape[-1])]
#         # calculate average inner and inter class distances
#         if self.num_classes > 0:
#             clabel = labels[::self.num_instances]   # label of each class
#             dd_inner = d_inner.mean(-1).reshape(-1,self.num_instances)
#             dd_inter = d_inter.mean(-1).reshape(-1,self.num_instances)
#             new_dist = torch.stack([dd_inner.detach(), dd_inter.detach()], dim=-1).mean(1)     # new average dist
#             # 更新参考距离
#             self.dist_bank[clabel, :2] = 0.9 * self.dist_bank[clabel, :2] + (1 - 0.9) * new_dist
#             dg_ap = self.dist_bank[labels, 0].detach()  # distbank + 平均参考距离
#             dg_an = self.dist_bank[labels, 1].detach()
#         # dg_ap = d_inner.mean(-1).detach()   # 平均参考距离
#         # dg_an = d_inter.mean(-1).detach()
#         # dg_ap = d_inner[..., 0].detach()   # 最大值作为参考距离
#         # dg_an = d_inter[..., 0].detach()
#         r_distp = d_inner.mean() / global_feat.norm(dim=-1).mean()
#         r_distn = d_inter.mean() / global_feat.norm(dim=-1).mean()
#         # batch hard triplet sample
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         # dist_an = dist_an.detach()
#
#         if dg_an is None:
#             dg_ap = dist_ap.detach()    # referance distance
#             dg_an = dist_an.detach()
#         mn = (dist_an/(dg_ap + dist_an)).detach()
#         mp = (dist_ap/(dist_ap + dg_an)).detach()
#         self.count += 1
#         if self.count >= 20:
#             self.count = 0
#             # self.logger.info('mp:{:.3f}({:.3f}/{:.3f}) mn:{:.3f}({:.3f}/{:.3f}) rdist:{:.3f}/{:.3f}'.format(
#             #     mp.mean(), mp.min(), mp.max(), mn.mean(), mn.min(), mn.max(), r_distp, r_distn))
#             self.logger.info('rd:{:.3f}/{:.3f} eft:{}'.format(
#                 r_distp, r_distn, (dist_an/(dg_ap + dist_an) + self.alpha - dist_ap/(dist_ap + dg_an) < self.margin).sum().item()))
#             # self.avg_rdist.flatten(), self.avg_rdist[2] - self.avg_rdist[1]))
#         self.rdist = (dist_ap/(dist_ap + dg_an)).detach()
#         # if self.margin == 0:
#         #     loss = self.softplus(mn.mean().clamp(0.5 + self.alpha, 1) - dist_an/(dg_ap + dist_an)) \
#         #            + self.softplus(dist_ap/(dist_ap + dg_an) - mp.mean().clamp(0, .5 - self.alpha))
#         #     loss = 2 * loss.mean()
#         # elif self.margin is not None:
#         # loss = (self.ranking_loss(dist_an/(dg_ap + dist_an) + self.alpha, dist_ap/(dist_ap + dg_an), y))
#         loss_p = dist_ap/(dist_ap + dg_an)
#         loss_n = 1 - dist_an/(dg_ap + dist_an)
#         if self.sigma < 1e-6:
#             loss_n = loss_n.detach()
#         loss = loss_p * ((self.p*loss_p).pow(self.beta).clamp(0, 1).detach()) \
#                + loss_n * ((self.p*loss_n).pow(self.beta).clamp(0, 1).detach())
#         loss = loss.mean()
#         # else:
#         #     loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an
#
#
# class ADMarginRelativeTripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None, num_classes=0, num_instances=4, alpha=0.0, beta=0.9, p=1.0, sigma=100.0, gamma=0
#                  , normalize_feature=False):
#         self.margin = margin
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.normalize_feature = normalize_feature
#         self.ad_margin = [0, 0]
#         self.num_instances = num_instances
#         self.dist_bank = None     # 记录每个class的平均类内距离和类间距离
#         self.avg_rdist = None
#         self.p = p
#         self.sigma = sigma
#         self.softplus = torch.nn.Softplus(50)
#         self.logger = logging.getLogger("reid_baseline.train")
#         self.count = 0
#         self.rdist = 0
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=min(margin, 1))
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if self.avg_rdist is None:
#             self.avg_rdist = torch.ones(3, 2, device=global_feat.device)
#         if self.normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dg_ap = None    # referance distance
#         dg_an = None
#         # calculate average inner and inter class distances
#         # if self.num_classes > 0:
#         mask = labels.expand(*labels.shape, *labels.shape) == labels.expand(*labels.shape, *labels.shape).t()  # mask = mask.float()
#         org_dist = dist_mat.detach()
#         d_inner = dist_mat[mask & ~torch.eye(mask.shape[0], dtype=torch.bool, device=labels.device)] \
#             .detach().reshape(mask.shape[0], -1)
#         d_inter = dist_mat[torch.logical_not(mask)].detach().reshape(mask.shape[0], -1)
#
#         d_inner = d_inner.sort(-1, True)[0]     # [:, :math.ceil(self.p * d_inner.shape[-1])]
#         d_inter = d_inter.sort(-1, False)[0]        # [:, :math.ceil(self.p * d_inter.shape[-1])]
#         dg_ap = d_inner.mean(-1).detach()   # 平均参考距离
#         dg_an = d_inter.mean(-1).detach()
#         # dg_ap = d_inner[..., 0].detach()   # 最大值作为参考距离
#         # dg_an = d_inter[..., 0].detach()
#         r_distp = d_inner.mean() / global_feat.norm(dim=-1).mean()
#         r_distn = d_inter.mean() / global_feat.norm(dim=-1).mean()
#         # batch hard triplet sample
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         # dist_an = dist_an.detach()
#
#         if dg_an is None:
#             dg_ap = dist_ap.detach()    # referance distance
#             dg_an = dist_an.detach()
#         mn = (dist_an/(dg_ap + dist_an)).detach()
#         mp = (dist_ap/(dist_ap + dg_an)).detach()
#         self.rdist = (dist_ap/(dist_ap + dg_an)).detach()
#
#         if self.normalize_feature:      # 归一化triplet
#             dist = dist_an - dist_ap
#             # loss = self.ranking_loss(dist_an, dist_ap, y)
#         elif self.margin is not None:   # relative triplet
#             dist = dist_an/(dg_ap + dist_an) - dist_ap/(dist_ap + dg_an)
#             # loss = (self.ranking_loss(dist_an/(dg_ap + dist_an) + self.alpha, dist_ap/(dist_ap + dg_an), y))
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         if self.margin > 1:
#             dist = dist_an - dist_ap
#             # loss = torch.clamp_min((self.margin - 1) * dist_ap.detach() - dist, 0)
#             loss = torch.clamp_min(torch.clamp_min(
#                 (dist_an / dist_ap - 1).mean().detach(), (self.margin - 1)) * dist_ap.detach() - dist, 0)
#         else:
#             loss = torch.clamp_min(torch.clamp_min(dist.mean().detach(), self.margin) - dist, 0)
#         self.count += 1
#         if self.count >= 20:
#             self.count = 0
#             # self.logger.info('mp:{:.3f}({:.3f}/{:.3f}) mn:{:.3f}({:.3f}/{:.3f}) rdist:{:.3f}/{:.3f}'.format(
#             #     mp.mean(), mp.min(), mp.max(), mn.mean(), mn.min(), mn.max(), r_distp, r_distn))
#             self.logger.info('rd:{:.3f}/{:.3f} eft:{}'.format(
#                 r_distp, r_distn, (loss != 0).sum().item()))
#             # self.avg_rdist.flatten(), self.avg_rdist[2] - self.avg_rdist[1]))
#
#         return loss.mean(), dist_ap, dist_an
#

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
        # if self.use_gpu: targets = targets.cuda()
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


# 占位符
class NoneCLS(nn.Module):
    def forward(self, x, *args):
        return 0 * x.sum()


class NoneTri(nn.Module):
    def forward(self, x, *args):
        return 0 * x.sum(), 0, 0


class CrossEntropyLabelSmoothwithMargin(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmoothwithMargin, self).__init__()
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