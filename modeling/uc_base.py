import torch
from torch import nn
import torch.nn.functional as F
from .cls_base import CLSBaseline
from .tri_base import Baseline, BaselineBN, BaselineCT, BaselineBN2d


# x, x2, x3
# l1
# l2
def Legendre(n):
    c = torch.eye(n+1)
    for i in range(1, n):
        c[i+1] = -i/(i+1)*c[i-1]
        c[i+1][1:] += (2*i+1)/(i+1)*c[i][:-1]
    return c

#
class Norm2UC(nn.Module):
    def __init__(self, n_dim=32):
        super(Norm2UC, self).__init__()
        self.n_dim = n_dim
        self.uc_k = nn.Linear(1, n_dim)
        self.uc_kx = nn.Parameter(torch.zeros(1))
        self.uc_kout = nn.Linear(n_dim, 1, bias=False)
        self.relu = nn.LeakyReLU(0.1)

        # fixed base
        self.uc_k.weight.requires_grad_(False)
        nn.init.constant_(self.uc_k.weight, 1)      # 初始化的影响？
        self.uc_k.bias.requires_grad_(False)
        nn.init.constant_(self.uc_k.bias, 0)
        self.uc_k.bias += torch.linspace(-0.8, 0.8, n_dim)  # (-1.0, 1.0, n_dim)
        # learned base
        # nn.init.normal_(self.uc_k.weight, mean=0e-2, std=1e-2)      # 初始化的影响？
        # nn.init.normal_(self.uc_k.bias, mean=0, std=1e-2)
        nn.init.normal_(self.uc_kout.weight, mean=0e-1 / n_dim, std=1e-4/(n_dim**0.5))
        # nn.init.constant_(self.out.weight, 0.1)
        # nn.init.constant_(self.out.bias, 0)
        # nn.init.constant_(self.out.bias, 0)

    def forward(self, norm):
        norm = (0.1*norm).clamp(-0.4, 0.4)  # (-1.0, 0.5)#.detach()
        norm = 0.1 * norm + 0.9 * norm.detach()
        out = norm * self.uc_kx
        if norm.dim() == 1:
            norm = norm.unsqueeze(-1)
        x = self.uc_k(norm)
        x = self.relu(x)
        out = self.uc_kout(x).squeeze()
        return out * ((self.n_dim) ** (-0.5))


class UCBase(nn.Module):
    taylor_n = 5

    def __init__(self, *args, fbase='tlr', renorm=True, **kwargs):
        super().__init__(**kwargs)
        self.renorm = renorm
        self.odd = 2        # 是否使用偶次项
        self.norm2uc=nn.Sequential()
        self.bn_norm = nn.BatchNorm1d(1)
        nn.init.constant_(self.bn_norm.weight, 1)     # 静态归一化
        self.bn_norm.weight.requires_grad_(False)
        self.bn_norm.bias.requires_grad_(False)
        if 'all' in fbase:
            self.odd = 1
        elif 'nn' in fbase:
            try:
                n_hidden = int(fbase.split('_')[-1])
            except ValueError as e:
                n_hidden = 32
            self.norm2uc = Norm2UC(n_hidden)    # margin Assingment module
        self.fbase = fbase.split('_')[0]
        self.log_norm = False
        if 'log' in self.fbase:
            self.fbase = self.fbase.replace('log', '')
            self.log_norm = True
        try:
            self.taylor_n = (int(fbase.split('_')[-1])+1) // self.odd
        except ValueError as e:
            print(e)
        self.uc_k = nn.Parameter(torch.zeros(self.taylor_n * (3 - self.odd)))
        # self.uc_k = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('avg_norm', torch.tensor(0.0))
        self.register_buffer('avg_uc', torch.tensor(0.0))
        taylor_const = torch.arange(self.taylor_n) + 1.0
        # self.register_buffer('taylor_const', 1 / taylor_const)  # 1/n!
        self.register_buffer('legendre_const', Legendre(self.taylor_n*2)[1::self.odd, 1::self.odd])  # 1/n!



    def get_uc(self, feats):
        f_norm = feats.norm(dim=-1) # .log()   # 对norm使用对数刻度
        if self.log_norm:
            f_norm = f_norm.log()
        batch_avg_norm = f_norm.mean()

        x = self.bn_norm(f_norm[:, None])[..., 0]

        x_pow = (x.expand(self.taylor_n*2, -1).t()).cumprod(dim=-1)[:, ::self.odd]
        if self.fbase == 'lgd':
            # 勒让德多项式
            factors = (self.legendre_const * self.uc_k[:, None]).sum(0)
            uc_r = 1 * (factors * x_pow).sum(-1) # / self.taylor_n / 2
        elif self.fbase == 'tlr':
            # 泰勒展开
            uc_r = 1 * (self.uc_k * x_pow).sum(-1) # / self.taylor_n / 2
        elif self.fbase == 'nn':
            # 泰勒展开
            uc_r = self.norm2uc(x)  # .exp() - 1
        else:
            raise KeyError(self.fbase)

        uc_shift = uc_r.clamp(-0.5, 0.5).mean()
        if not self.renorm:
            uc_shift = 0*uc_shift
        # uc_r = 0.1 * self.uc_k * x
        if self.training:
            # margin rebalance module
            self.avg_uc += 0.01*(uc_shift.detach() - self.avg_uc)
            return uc_r - uc_shift
        else:
            return uc_r - self.avg_uc


class UcTri_(UCBase):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def forward(self, x):
        global_feat = self.feature(x)

        uc_r = self.get_uc(global_feat)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            if self.neck == 'norm':
                cls_score = F.linear(F.normalize(feat), F.normalize(self.classifier.weight))
            else:
                cls_score = self.classifier(feat)
            if self.neck == 'no-bn':        # bn 后特征算tri
                global_feat = feat
            return cls_score, global_feat[:, ::1], uc_r  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                feat = F.normalize(feat) * global_feat.norm(dim=-1, keepdim=True)
                return feat, uc_r
            else:
                # print("Test with feature before BN")
                return global_feat, uc_r


class UcCls_(UCBase):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
        # print(" ")
        # UCBase.__init__(self)
        # self.uc_k = nn.Parameter(torch.tensor(0.1))
        # self.register_buffer('avg_norm', torch.tensor(20.0))

    def forward(self, x):
        feat = self.feature(x)
        feat = self.bottleneck(feat)
        # feat = self.scale(feat)

        f_norm = feat.norm(dim=-1)
        w_norm = self.classifier.weight.norm(dim=-1)

        # batch_avg_norm = f_norm.mean()
        # self.avg_norm = 0.1*(batch_avg_norm.detach() - self.avg_norm)
        # norm_base = batch_avg_norm + batch_avg_norm.detach() - self.avg_norm
        # uc_r = self.uc_k * (f_norm - norm_base) / norm_base
        uc_r = self.get_uc(feat)

        if self.training:
            # cls_score = self.classifier(feat)
            # cos_sim = cls_score/f_norm[:, None]/w_norm
            cos_sim = F.linear(F.normalize(feat), F.normalize(self.classifier.weight))
            # cos_sim = self.classifier(feat)
            return cos_sim, f_norm, w_norm * 1, uc_r    # 减少weight decay影响
        else:
            return feat, uc_r


class UcTri(Baseline, UcTri_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super(UcTri_, self).__init__(*args, **kwargs)

    def forward(self, x):
        return UcTri_.forward(self, x)


class UcTriBN(BaselineBN, UcTri_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super(UcTri_, self).__init__(*args, **kwargs)

    def forward(self, x):
        return UcTri_.forward(self, x)

class UcTriBN2d(BaselineBN2d, UcTri_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super(UcTri_, self).__init__(*args, **kwargs)

    def forward(self, x):
        return UcTri_.forward(self, x)

class UcTriCT(BaselineCT, UcTri_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super(UcTri_, self).__init__(*args, **kwargs)

    def forward(self, x):
        return UcTri_.forward(self, x)


class UcCls(CLSBaseline, UcCls_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CLSBaseline.__init__(self, *args, **kwargs)
        # UcCls_.__init__(self, *args, **kwargs)
        # pass
        # super(UcCls_, self).__init__(**kwargs)
        # super(CLSBaseline, self).__init__(*args, **kwargs)

    def forward(self, x):
        return UcCls_.forward(self, x)

# if __name__ == '__main__':
#     import numpy as np
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     self.uc_k[:] = torch.tensor([-0.1834, 0.1384, -0.0710, 0.0043, 0.0443])
#     factors = (self.legendre_const * self.uc_k[:, None]).sum(0)
#     uc_r = 1 * (factors * x_pow).sum(-1)
#     plt.plot(x.detach().numpy(), uc_r.detach().numpy())
#     fig.savefig('uc.png', transparent=False, dpi=300, bbox_inches="tight")