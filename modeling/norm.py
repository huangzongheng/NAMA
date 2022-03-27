import torch
from torch import nn
import torch.nn.functional as F


# 训练也使用buffer中的值
class OfflineBN(nn.BatchNorm1d):

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        # 更新buffer
        if bn_training:
            _ = F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                None, None, bn_training, exponential_average_factor, self.eps)

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean,
            self.running_var,
            self.weight, self.bias, False, exponential_average_factor, self.eps)


class Centering(nn.Module):
    def __init__(self, num_features, momentum=0.1, shift=False, scale=False):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('var', torch.zeros(num_features))
        self.shift = shift
        self.scale = scale
        if self.shift:
            self.bias = nn.Parameter(torch.zeros(num_features))
        if self.scale:
            self.weight = nn.Parameter(torch.ones(num_features))
            nn.init.constant_(self.weight, 2)

    def forward(self, x):
        if self.training:
            running_mean = x.mean(0)
            running_var = x.std(0).clamp(1e-5, 10)
            self.running_mean += self.momentum * (running_mean.detach() - self.running_mean)
            self.var += self.momentum * (running_var.detach() - self.var)
        else:
            running_mean = self.running_mean
            running_var = self.var
        out = x - running_mean #
        if self.shift:
            out = (out + self.bias * running_var)
        if self.scale:
            out = out * self.weight
        return out

# 所有channel使用相同缩放系数
class Scale(nn.Module):
    def __init__(self, num_features=0, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer('var', torch.zeros(num_features))
        self.weight = nn.Parameter(torch.ones(1))   # 1
        nn.init.constant_(self.weight, 1.6)
        # self.register_parameter('weight', torch.ones(num_features))

    def forward(self, x):
        return x * self.weight

# 每个channel分别scale
# class Scale(nn.Module):
#     def __init__(self, num_features, momentum=0.1):
#         super().__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.register_buffer('var', torch.zeros(num_features))
#         self.weight = nn.Parameter(torch.ones(num_features))
#         # self.register_parameter('weight', torch.ones(num_features))
#
#     def forward(self, x):
#         n = x.shape[0]
#         if self.training:
#             # running_var = ((x - x.mean(0, keepdim=True)).pow(2).mean(0) + 1e-5).sqrt()
#             running_var = x.std(0).clamp(1e-5, 10)
#             self.var += self.momentum * (running_var - self.var)
#         else:
#             running_var = self.var

        return (x / running_var) * self.weight



if __name__ == '__main__':
    ct = Centering(5)
    scale = Scale(5)
    bn = nn.BatchNorm1d(5, eps=1e-12)
    fbn = OfflineBN(5, track_running_stats=True)
    x = torch.rand((80, 5))
    print(bn(x).mean(0))
    print(fbn(x).mean(0))
    print(ct(x).mean())
    print(scale(ct(x)).mean())
