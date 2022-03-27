import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math
import pdb
import numpy as np

class CosFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m

    # def forward(self, similarity, fn, wn, label):
    def forward(self, similarity, fn, wn, uc, label):

        correct = ((similarity.max(1)[1] == label).float().mean())
        similarity = similarity.clamp_min(0)
        s = self.s # ** correct
        one_hot = torch.zeros_like(similarity)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        logits = s * (similarity - one_hot * self.m)
        loss = F.cross_entropy(logits, label)
        # 加入正则约束后，极大的提升了性能，mAP提升2~3
        # loss = loss + 0.01 * (wn - 10).pow(2).mean() + 0.01 * (fn - 40).pow(2).mean()
        return loss

class NormFace(nn.Module):
    def __init__(self, s=16.0, m=0.0):
        super().__init__()
        self.s = s
        # self.m = m

    def forward(self, similarity, label):

        # similarity = similarity.clamp_min(0)
        one_hot = torch.zeros_like(similarity)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        logits = self.s * similarity
        loss = F.cross_entropy(logits, label)
        return loss


class CosfacePairwiseLoss(nn.Module):
    def __init__(self, s=30.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m

    @staticmethod
    def get_sim(f1, f2, dim=-1):
        return torch.cosine_similarity(f1, f2, dim=dim)

    def forward(self, feat, label):

        mask = label[:, None].eq(label)
        similarity = self.get_sim(feat[:, None], feat, dim=-1)
        similarity = similarity - mask * self.m
        s_pos = - similarity * mask - 10 * ~mask
        s_neg = similarity * ~mask - 10 * mask

        logit_p = torch.logsumexp(s_pos*self.s, dim=-1)
        logit_n = torch.logsumexp(s_neg*self.s, dim=-1)
        loss = F.softplus(logit_p + logit_n).mean()

        return loss, 0, 0


class ArcFaceLoss(CosFaceLoss):

    def forward(self, similarity, fn, wn, uc, label):

        correct = ((similarity.max(1)[1] == label).float().mean())
        similarity = similarity.clamp_min(0)
        similarity = torch.acos(similarity.clamp(-0.99, 0.99))
        s = self.s # ** correct
        one_hot = torch.zeros_like(similarity)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        logits = s * torch.cos(similarity + one_hot * self.m)
        loss = F.cross_entropy(logits, label)
        # 加入正则约束后，极大的提升了性能，mAP提升2~3
        # loss = loss + 0.01 * (wn - 10).pow(2).mean() + 0.01 * (fn - 40).pow(2).mean()
        return loss

    # def forward(self, similarity, fn, wn, uc, label):
    # # def forward(self, similarity, fn, wn, label):
    #     similarity = 1.57 - torch.acos(similarity.clamp(-0.99, 0.99))
    #     return super().forward(similarity, fn, wn, uc, label)
        # return super().forward(similarity, fn, wn, label)

class ArcFacePairwiseLoss(CosfacePairwiseLoss):
    @staticmethod
    def get_sim(f1, f2, dim=-1):
        similarity = -torch.cosine_similarity(f1, f2, dim=dim)
        return torch.acos(similarity.clamp(-0.99, 0.99))

# 正样本距离用cos，负样本距离用sin
class ProjCLSLoss(nn.Module):
    def __init__(self, s=30.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, cos_sim, fn, wn, label):

        correct = ((cos_sim.max(1)[1] == label).float().mean())
        # angle = torch.acos(cos_sim.clamp(-0.99, 0.99))
        sin_sim = 1 - torch.sqrt(1 - cos_sim.clamp(-0.99, 0.99).pow(2))
        s = self.s # ** correct
        one_hot = torch.zeros_like(cos_sim)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        similarity = one_hot * cos_sim + (1 - one_hot) * sin_sim

        logits = s * (similarity - one_hot * self.m)
        loss = F.cross_entropy(logits, label)
        return loss + 0.01 * (wn - 10).pow(2).mean() + 0.01 * (fn - 40).pow(2).mean()



# def LogSumExp(score, mask):
#     max_score = score.max()
#     max_score = max_score.unsqueeze(0).unsqueeze(1).expand_as(score)
#     score = score - max_score * (1-mask)   # elimintate the scores which are of none use
#     max_score, _ = score.max(1)
#     max_score_reduce = max_score.unsqueeze(1).expand_as(score)
#     score = score - max_score_reduce
#     return max_score + ((score.exp() * mask).sum(1)).log()
# #
# #
# class CosfacePairwiseLoss(nn.Module):
#     def __init__(self, m=0.2, s=16):
#         super(CosfacePairwiseLoss, self).__init__()
#         self.m = m
#         self.s = s
#         self.simi_pos = None
#         self.simi_neg = None
#
#     def __call__(self, input, target):
#         input = F.normalize(input)
#         # pdb.set_trace()
#         n = input.size(0)
#         # target = target.cuda()
#         mask = target.expand(n, n).eq(target.expand(n, n).t())
#         mask = mask.float()
#         mask_self = torch.FloatTensor(np.eye(n)).to(input.device)
#         mask_pos = mask - mask_self
#         mask_neg = 1 - mask
#
#         simi = input.mm(input.t())
#         self.simi_pos = LogSumExp(- simi * self.s, mask_pos).mean() / (- self.s)
#         self.simi_neg = LogSumExp(simi * self.s, mask_neg).mean() / self.s
#         simi = (simi - self.m * mask) * self.s
#
#         # '''
#         pos_LSE_cmp = LogSumExp(- simi, mask_pos)
#         neg_LSE_cmp = LogSumExp(simi, mask_neg)
#
#         loss_cmp = F.softplus(pos_LSE_cmp + neg_LSE_cmp)
#         # '''
#
#         '''
#         mask_pos, mask_neg = mask_pos.bool(), mask_neg.bool()
#         pos_pairs = torch.masked_select(simi, mask_pos).reshape(n, -1)
#         neg_pairs = torch.masked_select(simi, mask_neg).reshape(n, -1)
#         pos_LSE = torch.logsumexp(- pos_pairs, 1)
#         neg_LSE = torch.logsumexp(neg_pairs, 1)
#         loss = F.softplus(pos_LSE + neg_LSE)
#         '''
#
#         return loss_cmp.mean(), 0, 0


if __name__ == '__main__':
    torch.random.manual_seed(0)
    cf = CosfacePairwiseLoss(m=0.3, s=30)
    x = torch.rand(8, 200)
    x[:4] += 0.5
    x[4:] -= 0.5
    label = torch.tensor([0,0,0,0,1,1,1,1])
    l = cf(x, label)
    print(l)


# class CosfacePairwiseLossqg(nn.Module):
#     def __init__(self, m=0.2, s=16):
#         super(CosfacePairwiseLossqg, self).__init__()
#         self.m = m
#         self.s = s
#         self.simi_pos = None
#         self.simi_neg = None
#
#     def __call__(self, feat, attention, target):
#         # input = F.normalize(input)
#         # pdb.set_trace()
#         n = target.size(0)
#         # target = target.cuda()
#         mask = target.expand(n, n).eq(target.expand(n, n).t())
#         mask = mask.float()
#         mask_self = torch.FloatTensor(np.eye(n)).to(target.device)
#         mask_pos = mask - mask_self
#         mask_neg = 1 - mask
#
#         # simi = input.mm(input.t())
#         if isinstance(feat, torch.Tensor):
#             simi = get_qg_dist(feat, attention, feat, attention, 'cos')
#         elif isinstance(feat, (list, tuple)):
#             simi = get_qg_dist(feat[0], attention[0], feat[1], attention[1], 'cos')
#         # simi = get_qg_dist(feat, attention, feat, attention, 'cos')
#         self.simi_pos = LogSumExp(- simi * self.s, mask_pos).mean() / (- self.s)
#         self.simi_neg = LogSumExp(simi * self.s, mask_neg).mean() / self.s
#         loss_dig = 1 - simi.diag()
#         simi = (simi - self.m * mask) * self.s
#
#         # '''
#         pos_LSE_cmp = LogSumExp(- simi, mask_pos)
#         neg_LSE_cmp = LogSumExp(simi, mask_neg)
#
#         loss_cmp = F.softplus(pos_LSE_cmp + neg_LSE_cmp)
#         # '''
#
#         '''
#         mask_pos, mask_neg = mask_pos.bool(), mask_neg.bool()
#         pos_pairs = torch.masked_select(simi, mask_pos).reshape(n, -1)
#         neg_pairs = torch.masked_select(simi, mask_neg).reshape(n, -1)
#         pos_LSE = torch.logsumexp(- pos_pairs, 1)
#         neg_LSE = torch.logsumexp(neg_pairs, 1)
#         loss = F.softplus(pos_LSE + neg_LSE)
#         '''
#         loss = loss_cmp.mean() # + loss_dig.mean()
#
#         return loss, 0, 0
