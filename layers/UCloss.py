import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import logging


class BaseUCLoss(nn.Module):

    def __init__(self, margin=0, lm=0.05, um=0.25, lambda_g=1.0, la=10, ua=110, reg_type='exp_3', shift_margin=False,
                 normalize=True, uc=True, metric='arc', *args, **kwargs):
        super().__init__()
        self.avg_m = 0
        self.min_m = 0
        self.max_m = 0
        self.avg_l = 0
        self.margin = margin
        self.lm = lm
        self.um = um
        self.la = la
        self.ua = ua
        self.lambda_g = lambda_g
        r = reg_type.split('_')
        self.reg = r[0]
        self.k = float(r[1]) if len(r) > 1 else 3
        self.shift_margin = shift_margin
        self.normalize = normalize
        self.use_uc = uc
        self.metric = metric

    @staticmethod
    def get_dist(feat1, feat2):     # 默认弧度距离
        # return euclidean_dist(feat1, feat2)
        # similarity = -torch.cosine_similarity(feat1, feat2, dim=-1)
        # return torch.acos(similarity.clamp(-0.99, 0.99))
        dist = torch.cdist(feat1, feat2, compute_mode='donot_use_mm_for_euclid_dist')
        return torch.asin(dist / 2) * 2

    def m(self, norms):
        norms = norms.clamp(self.la, self.ua)
        x = (norms - self.la) / (self.ua - self.la)
        margin = (self.um - self.lm) * x + self.lm
        return margin

    def g(self, norms: torch.Tensor):

        normed_x = ((norms - self.la) / (self.ua - self.la))    # la:0, ua:1
        if self.reg == 'exp':
            # normed_x = (normed_x)
            reg = torch.exp(-normed_x * self.k) / self.k
            ug = 2.7183 ** -self.k
            reg = (reg + ug * normed_x) / (1 - ug)
            # reg = (torch.exp(-normed_x) + normed_x / (2.7183 ** self.k))/(self.k*(1 - 2.7183 ** -self.k))
        elif self.reg == 'quad':    # x^2
            reg = (1-normed_x).pow(2) / 2
        elif self.reg == 'log':
            normed_x = (norms.clamp_min(self.la) / (self.ua - self.la))
            ug = 1 / (self.ua / (self.ua - self.la))
            lg = 1 / (self.la / (self.ua - self.la))
            reg = -torch.log(normed_x)
            reg = (reg + ug * normed_x) / (lg - ug)
            # reg = -torch.log(norms.clamp(1, self.ua)/self.la)
        elif self.reg == 'arc':     # 1/x
            normed_x = (norms.clamp_min(self.la) / (self.ua - self.la))
            ug = 1 / (self.ua / (self.ua - self.la))**2
            lg = 1 / (self.la / (self.ua - self.la))**2
            reg = 1 / normed_x
            reg = (reg + ug * normed_x) / (lg - ug)
        # elif self.reg == 'linear':
        #     pass
        else:
            raise NotImplementedError(self.reg)
        reg = reg.mean()

        return reg


class UCCLSLoss(BaseUCLoss):
    def __init__(self, *args, s=30, **kwargs):
        super().__init__(*args, **kwargs)
        # self.avg_lf = 0
        self.avg_lw = 0
        self.s = s

    def compute_loss(self, similarity, fn, wn, uc, label):
        # mf = uc
        mf = self.m(fn)
        mw = self.m(wn) * 0
        reg = ((self.g(fn) + self.g(wn[label])) / 2) * self.lambda_g # * self.s

        one_hot = torch.zeros_like(similarity)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        dist = torch.acos(similarity.clamp(-0.99, 0.99))      # arc dist
        dist = dist + (1 - one_hot) * (mw + mf[:, None]) - one_hot * (mw + mf[:, None])

        # if not mf.mean().isnan():
        self.avg_m += 0.1 * (self.margin*0 - 2*(mf.mean() + mw.mean()).detach().item()- self.avg_m)
        self.min_m += 0.1 * (self.margin*0 - 2*(mf.min() + mw.mean()).detach().item() - self.min_m)
        self.max_m += 0.1 * (self.margin*0 - 2*(mf.max() + mw.mean()).detach().item() - self.max_m)
        self.avg_l += 0.1 * (fn.mean().detach().item() - self.avg_l)
        self.avg_lw += 0.1 * (wn.mean().detach().item() - self.avg_lw)
        # else:
        #     print('nan margin')
        # self.avg_l = (self.avg_lf, self.avg_lw)

        logits = - self.s * (dist + one_hot * (self.margin)) # - self.avg_m
        loss = F.cross_entropy(logits, label)
        return loss, reg # + 1e-2 * (fn - 25).pow(2).mean()

    def forward(self, similarity, fn, wn, label):
        loss, reg = self.compute_loss(similarity, fn, wn, label)
        return loss + reg - reg.detach()


class UCShiftCLSLoss(BaseUCLoss):
    def __init__(self, *args, s=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = s

    @staticmethod
    def get_dist(feat1, feat2):
        # return euclidean_dist(feat1, feat2)
        similarity = -torch.cosine_similarity(feat1, feat2, dim=-1)
        return torch.acos(similarity.clamp(-0.99, 0.99))

        # dist = torch.cdist(feat1, feat2, compute_mode='donot_use_mm_for_euclid_dist')
        # return torch.asin(dist / 2) * 2

    def compute_loss(self, similarity, fn, wn, uc, label):
        mf = uc
        reg = (self.g(fn)) * self.lambda_g # * self.s

        one_hot = torch.zeros_like(similarity)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        if self.metric == 'arc':
            dist = -torch.acos(similarity.clamp(-0.99, 0.99))      # arc dist
        elif self.metric == 'cos':
            dist = (similarity.clamp(-0.99, 0.99))      # arc dist
        else:
            dist = 1.57 - torch.acos(similarity.clamp(-0.99, 0.99))      # arc sim

        if self.use_uc:
            dist = dist + (1 - one_hot) * (mf[:, None]) - one_hot * (mf[:, None])

        self.avg_m += 0.1 * (self.margin - 2*(mf.mean()).detach().item()- self.avg_m)
        self.min_m += 0.1 * (self.margin - 2*(mf.min()).detach().item() - self.min_m)
        self.max_m += 0.1 * (self.margin - 2*(mf.max()).detach().item() - self.max_m)
        self.avg_l += 0.1 * (fn.mean().detach().item() - self.avg_l)

        # 标准arcface
        # dist = -torch.cos(dist)

        # logits = - self.s * (dist + one_hot * (self.margin)) # - self.avg_m
        if self.metric == 'arc':
            # dist = torch.acos(similarity.clamp(-0.99, 0.99))      # arc dist
            logits = self.s * torch.cos(-dist + one_hot * (self.margin)) # - self.avg_m
        elif self.metric == 'cos':
            logits = self.s * (dist + one_hot * (self.margin)) # - self.avg_m
        else:
            logits = self.s * (dist + one_hot * (self.margin)) # - self.avg_m
        loss = F.cross_entropy(logits, label)
        # reg = 0.0 * (wn - 10).pow(2).mean() + 0.01 * (fn - 40).pow(2).mean()
        return loss, reg # + 1e-2 * (fn - 25).pow(2).mean()

    def forward(self, similarity, fn, wn, uc, label):
        loss, reg = self.compute_loss(similarity, fn, wn, uc, label)
        return loss + reg - reg.detach()



# 去除S对梯度绝对值的影响
class UCNormedCLSLoss(UCCLSLoss):

    def forward(self, similarity, fn, wn, label):
        loss, reg = self.compute_loss(similarity, fn, wn, label)
        return loss/self.s + reg - reg.detach()

# softmax loss（监测模长）
class UCXentLoss(UCCLSLoss):

    def forward(self, similarity, fn, wn, label):
        self.avg_l += 0.1 * (fn.mean().detach().item() - self.avg_l)
        self.avg_lw += 0.1 * (wn.mean().detach().item() - self.avg_lw)
        similarity = similarity * fn[:, None] * wn
        loss = F.cross_entropy(similarity, label)
        # loss, reg = self.compute_loss(similarity, fn, wn, label)
        return loss

class UCPairwiseLoss(BaseUCLoss):

    def __init__(self, *args, s=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = s

    def compute_loss(self, feat, label):
        norms = feat.norm(dim=-1)
        # feat = feat / norms[:, None]
        feat = F.normalize(feat)
        uc = self.m(norms)
        reg = self.g(norms) * self.lambda_g # * self.s

        mask = label[:, None].eq(label)
        similarity = -self.get_dist(feat, feat)
        # similarity.diagonal().zero_()

        correction = - (mask * (uc[:, None] + uc[None, :] - self.margin)) + (~mask * (uc[:, None] + uc[None, :]))   # pos -m, neg +m
        similarity = similarity - correction # - mask * self.margin

        s_pos = - similarity * mask - 10 * ~mask
        s_neg = similarity * ~mask - 10 * mask

        logit_p = torch.logsumexp(s_pos*self.s, dim=-1)
        logit_n = torch.logsumexp(s_neg*self.s, dim=-1)
        loss = F.softplus(logit_p + logit_n).mean()

        self.avg_l += 0.1 * (norms.mean().detach().item() - self.avg_l)
        self.avg_m += 0.1 * (self.margin - 4*uc.mean().detach().item() - self.avg_m)
        self.min_m += 0.1 * (self.margin - 4*uc.min().detach().item() - self.min_m)
        self.max_m += 0.1 * (self.margin - 4*uc.max().detach().item() - self.max_m)

        return loss, reg

    def forward(self, feat, label):
        loss, reg = self.compute_loss(feat, label)
        return loss + reg - reg.detach(), 0, 0


class UCNormedPairwiseLoss(UCPairwiseLoss):

    def forward(self, feat, label):
        loss, reg = self.compute_loss(feat, label)
        return loss/self.s + reg - reg.detach(), 0, 0

# use arcdist and arc margin
class UCShiftPairwiseLoss(BaseUCLoss):

    def __init__(self, *args, s=30, detach_pn=False, soft_mining=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = s
        self.detach_pn = detach_pn
        self.soft_mining = soft_mining
        # hidden_margin = (2*np.log(120) - np.log(0.1)) / self.s      # batchsize = 120, softplus cut thr = 0.1
        # self.margin = self.margin - hidden_margin

    @staticmethod
    def get_dist(feat1, feat2):     # 默认弧度距离
        dist = torch.cdist(feat1, feat2, compute_mode='donot_use_mm_for_euclid_dist')
        return dist

    @staticmethod
    def convert_dist(dist):     # arc to balabala + margin
        return torch.asin(dist / 2) * 2

    @staticmethod
    def invert_dist(dist):      # balabala + margin to bala
        return dist

    def compute_loss(self, feat, uc, label):
        norms = feat.norm(dim=-1)
        # feat = feat / norms[:, None]
        # uc = self.m(norms)
        reg = self.g(norms) * self.lambda_g # * self.s

        mask = label[:, None].eq(label)
        if self.normalize:
            feat = F.normalize(feat)
            similarity = -self.get_dist(feat, feat)
        else:
            similarity = -torch.cdist(feat.float(), feat.float(), compute_mode='donot_use_mm_for_euclid_dist')
        # similarity.diagonal().zero_()

        # 只回传anchor的uc梯度
        if self.detach_pn:
            correction = - (mask * (uc[:, None] + uc[None, :].detach() - self.margin)) + (
                        ~mask * (uc[:, None] + uc[None, :].detach()))
        else:
            correction = - (mask * (uc[:, None] + uc[None, :] - self.margin)) + (
                        ~mask * (uc[:, None] + uc[None, :]))  # pos -m, neg +m
        similarity = self.convert_dist(similarity)
        similarity = similarity - correction # - mask * self.margin
        similarity = self.invert_dist(similarity)

        s_pos = - similarity * mask - 10 * ~mask
        s_neg = similarity * ~mask - 10 * mask

        if self.soft_mining:
            logit_p = torch.logsumexp(s_pos*self.s, dim=-1)
            logit_n = torch.logsumexp(s_neg*self.s, dim=-1)
            loss = F.softplus(logit_p + logit_n).mean()
        else:
            d_ap = s_pos.max(dim=-1)[0]
            d_an = s_neg.max(dim=-1)[0]
            loss = F.relu(d_ap + d_an).mean() * self.s
        m = 0
        # if self.margin == 0:        # 浮动margin
        #     m = (logit_p + logit_n).detach().mean()
        #     # m.sort()[0][m.shape[0]//2]
        #     loss = F.softplus(logit_p + logit_n - m, beta=0.125).mean()
        #     m = m / self.s

        self.avg_l += 0.1 * (norms.mean().detach().item() - self.avg_l)
        self.avg_m += 0.1 * (self.margin - 4*uc.mean().detach().item() - self.avg_m - m)
        self.min_m += 0.1 * (self.margin - 4*uc.min().detach().item() - self.min_m - m)
        self.max_m += 0.1 * (self.margin - 4*uc.max().detach().item() - self.max_m - m)

        return loss, reg

    def forward(self, feat, uc, label):
        loss, reg = self.compute_loss(feat, uc, label)
        return loss/self.s + (reg - reg.detach()), 0, 0


# L2 dist and L2 margin
class UCShiftTripletLoss(UCShiftPairwiseLoss):

    @staticmethod
    def convert_dist(dist):     # arc to balabala + margin
        return dist

    @staticmethod
    def invert_dist(dist):      # balabala + margin to bala
        return dist

# L2 dist and arc margin
class UCShiftArcTripletLoss(UCShiftPairwiseLoss):

    @staticmethod
    def convert_dist(dist):     # arc to balabala + margin
        return torch.asin(dist / 2) * 2

    @staticmethod
    def invert_dist(dist):      # balabala + margin to bala
        return torch.sin(dist / 2) * 2


class UCArcTripletLoss(BaseUCLoss):
    beta = 20

    def forward(self, feat, labels):
        norms = feat.norm(dim=-1)
        feat = feat / norms[:, None]
        uc = self.m(norms)
        reg = self.g(norms) * self.lambda_g

        dist_mat = self.get_dist(feat, feat)
        mask = labels[:, None].eq(labels)
        correction = - (mask * (uc[:, None] + uc[None, :])) + (~mask * (uc[:, None] + uc[None, :]))  # pos -m, neg +m
        dist_mat = dist_mat + correction

        dist_ap, p_inds = torch.max(dist_mat - ~mask * 10, 1)
        dist_an, n_inds = torch.min(dist_mat + mask * 10, 1)

        diff = (dist_ap - dist_an).reshape(-1, labels.shape[0]) + self.margin

        loss = F.softplus(diff, beta=self.beta).mean()

        self.avg_l += 0.1 * (norms.mean().detach().item() - self.avg_l)
        self.avg_m += 0.1 * (self.margin - 4*uc.mean().detach().item() - self.avg_m)
        self.min_m += 0.1 * (self.margin - 4*uc.min().detach().item() - self.min_m)
        self.max_m += 0.1 * (self.margin - 4*uc.max().detach().item() - self.max_m)

        return loss + reg - reg.detach(), 0, 0


class UCTripletLoss(UCArcTripletLoss):

    @staticmethod
    def get_dist(feat1, feat2):
        dist = torch.cdist(feat1, feat2, compute_mode='donot_use_mm_for_euclid_dist')
        return dist



if __name__ == '__main__':
    L = UCCLSLoss(n_classes=10, n_feats=256)
    sim = torch.rand(4, 10)
    fn = torch.rand(4)*40
    wn = torch.rand(10)*40
    label = torch.arange(4)
    print(L(sim, fn, wn, label))
    print(L)