import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Reduce, Rearrange
from torchvision.models import resnet18, resnet50
from utils import cum_matmul, GussianBlur

import math

# x: ..., H, W tensor
# offset: N (x, y) tensor
def affine_sample(x, offset, stride=1):         # N=1时可用， N>1时还有问题
    h, w = x.shape[-2:]

    # pad = offset.abs().ceil().int().tolist()
    pad_lu = offset.neg().max(0)[0].clamp_min(0).ceil().int()    # 左上
    pad_rd = offset.clamp_min(0).max(0)[0].ceil().int()+1    # 右下
    pad = rearrange([pad_lu, pad_rd], "b c -> (c b)")
    x = F.pad(x, pad.tolist())               # 左右上下
    offset += pad_lu            # 转换坐标
    start = offset.floor().int()
    end = offset.ceil().int()
    res = offset - start
    xo, xx, xy, xxy = get_offset_corner(x, start[:, 1], start[:, 0], h, w, stride)
    # xx = get_offset_corner(x, start[:, 1], end[:, 0], h, w)
    # xy = get_offset_corner(x, end[:, 1], start[:, 0], h, w)
    # xxy = get_offset_corner(x, end[:, 1], end[:, 0], h, w)

    # xo = x[..., start[1]:, start[0]:][..., :h:stride, :w:stride]
    # xx = x[..., start[1]:, end[0]:][..., :h:stride, :w:stride]
    # xy = x[..., end[1]:, start[0]:][..., ::stride, :w:stride]
    # xxy = x[..., end[1]:, end[0]:][..., :h:stride, :w:stride]
    res = rearrange(res, 'n b -> n b 1 1 1')

    out = xo * ((1-res[:, 0])*(1-res[:, 1]))\
          + xx * (res[:, 0]*(1-res[:, 1]))\
          + xy * ((1-res[:, 0])*res[:, 1])\
          + xxy * (res[:, 0]*res[:, 1])

    return out

# x: ..., H+pad, W+pad tensor
# offset(int): N (x, y) tensor
def get_offset_corner(x, start_x, start_y, h, w, stride):
    x_ind = (torch.arange(0, x.shape[-1], device=x.device).expand(x.shape[0], -1)
             + start_x.unsqueeze(-1)).clamp(0, x.shape[-1]-1)     # N W
    y_ind = (torch.arange(0, x.shape[-2], device=x.device).expand(x.shape[0], -1)
             + start_y.unsqueeze(-1)).clamp(0, x.shape[-2]-1)     # N H
    grid = x_ind.unsqueeze(-2) + y_ind.unsqueeze(-1) * x.shape[-1]
    out = torch.gather(rearrange(x, 'n c h w -> n c (h w)'), -1,
                       rearrange(grid, 'n h w -> n 1 (h w)').expand(-1, x.shape[1], -1))
    out = rearrange(out, 'n c (h w) -> n c h w', h=x.shape[-2], w=x.shape[-1])

    return out[..., :h:stride, :w:stride], out[..., :h:stride, 1:w+1:stride], \
           out[..., 1:h+1:stride, :w:stride], out[..., 1:h+1:stride, 1:w+1:stride]


# 从输入图像预测ratio，得到kernal的仿射变换参数
class AffinePredictor(nn.Module):

    def __init__(self, forward_times=2):
        super().__init__()
        cnn = resnet50(pretrained=True)
        cnn_a = AffineResNet(layers=[3, 4, 6, 3], pred_affine=False)
        cnn_a.load_state_dict(cnn.state_dict(), strict=False)
        cnn_a.layer2 = nn.Sequential()
        cnn_a.layer3 = nn.Sequential()
        cnn_a.layer4 = nn.Sequential()
        self.cnn = cnn_a
        # self.forward_times = forward_times
        self.affine = nn.Sequential(
            # cnn_a.conv1,
            # cnn_a.bn1,
            # cnn_a.relu,
            # cnn_a.maxpool,
            # cnn_a.layer1,     # 64  256
            # # cnn.layer2,     # 128
            # # cnn.layer3,     # 256
            # # cnn.layer4,     # 512
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 4),
            Rearrange('b (x y) -> b x y', x=2, y=2)     # y=3
        )
        nn.init.constant_(self.affine[-2].weight, 0)
        self.affine[-2].bias.data = torch.tensor([0.0, 0., 0., 0.0])  # x1, y1, x2, y2
        # self.affine[-2].bias.data = torch.tensor([1.0, 0., 0., 1.0])  # x1, y1, x2, y2
        # self.affine.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x, forward_times=2):
        # print(x.shape, x.device)
        affine_offsets = torch.eye(2, device=x.device).expand(x.shape[0], -1, -1).int()
        out = [affine_offsets * (torch.tensor(x.shape[2:][::-1])/torch.tensor([128, 256.])).to(x.device)]
        # for i in range(forward_times):
        #     ft, _ = self.cnn(x, affine_offsets)
        #     pred = self.affine(ft)
        #     pred.diagonal(dim1=-1, dim2=-2).exp_()   # xy方向scale取指数
        #     # affine_offsets = pred * affine_offsets
        #     affine_offsets = torch.matmul(pred, affine_offsets.float())
        #     out.append(pred)
        out = torch.stack(out, dim=1)   # n k 2 2
        # out = torch.tensor([[1,0],[0.,2]], device=x.device).expand(x.shape[0], -1, -1)
        return out

# 将输入图片按照卷积核的大小和仿射参数采样
class AffineKernal(object):

    # inputs: img(n c h w), affine_offsets(n, 2, 2)
    # outputs: sampled img(n c h/stride*k, w/stride*k)
    def __init__(
            self,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding_mode: str = 'zeros',
            last_affine=None      # 重用之前计算的映射值
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.real_stride = stride   # 与正常卷积等效的stride
        self.stride = kernel_size   # 为了继承nn.Conv2d中的方法, 这里的stride是实际进行卷积时使用的stride
        self.padding = (0, 0)
        self.zero_pad = nn.ZeroPad2d(kernel_size//2)
        # self.cur_affine = None      # 本次forward 计算的affine mat

    # def _get_conv_inputs(self, x, affine_offset):   # affine: n, 2, 2(x, y)
    #     n, c, h, w = x.shape
    #     out = []
    #     offset_base = torch.arange(self.kernel_size, device=x.device).reshape(-1, 1, 1, 1) - self.kernel_size//2   # [-1, 0, 1]...
    #     offset_base = offset_base * affine_offset   # k, n, 2, 2(x,y)  a*[-1, 0, 1]*scale
    #     # to save memory
    #     for i in range(self.kernel_size**2):
    #         j = i // self.kernel_size
    #         i = i % self.kernel_size
    #         offset = (offset_base[j, :, 1] + offset_base[i, :, 0])
    #         out.append(affine_sample(x, offset, self.real_stride))
    #
    #     out = rearrange(out, '(k k1) b c h w -> b c (h k) (w k1)', k=self.kernel_size, k1=self.kernel_size)
    #
    #     return out

    def _get_conv_inputs(self, x, affine_offset):   # affine: n, 2, 2(x, y)
        n, c, h, w = x.shape
        out = []
        # if affine_offset.dtype is not torch.int:    # 形变太小则使用普通卷积
        if True:    # 形变太小则使用普通卷积
            normal_flag = 0
            grid_base = F.affine_grid(theta=torch.tensor([[1.0, 0., 0.], [0., 1.0, 0.]], device=x.device).expand(1, 2, 3),
                                      size=[1, c, h, w])    # n, h, w, 2(x,y)
            grid_base = grid_base[:, ::self.real_stride, ::self.real_stride].contiguous()

            scale = 2 / torch.tensor([w, h], dtype=torch.float, device=x.device)   # convert pixel coordinate to [-1, 1] scale
            offset_base = torch.arange(self.kernel_size, device=x.device).reshape(-1, 1, 1, 1) - self.kernel_size//2   # [-1, 0, 1]...
            offset_base = offset_base * affine_offset   # k, n, 2, 2(x,y)  a*[-1, 0, 1]*scale
        else:
            normal_flag = 1
        '''
        # original realization
        offset_kernal = rearrange(offset_base[:, :, 1], "ky n b-> ky 1 n 1 1 b", b=2) \
            + rearrange(offset_base[:, :, 0], "kx n b-> 1 kx n 1 1 b", b=2)
        grid_kernal = offset_kernal*scale + grid_base

        for grid in rearrange(grid_kernal, 'ky kx n h w b -> (ky kx) n h w b', b=2):
            out.append(F.grid_sample(x, grid))
        '''
        # to save memory
        for i in range(self.kernel_size**2):
            j = i // self.kernel_size
            i = i % self.kernel_size
            if normal_flag == 0:
                grid = scale * (offset_base[j, :, 1] + offset_base[i, :, 0])
                grid = rearrange(grid, "n b-> n 1 1 b", b=2) + grid_base
                out.append(F.grid_sample(x, grid))
            else:
                out.append(self.zero_pad(x)[..., j:j+h:self.real_stride, i:i+w:self.real_stride])

        out = rearrange(out, '(k k1) b c h w -> b c (h k) (w k1)', k=self.kernel_size, k1=self.kernel_size)

        # print('x:', x.shape, x.device)
        # print('a:', affine_offset.shape, affine_offset.device)
        # print('o:', out.shape, out.device)

        return out

    # def forward(self, x, affine_offsets):
    #     return self._get_conv_inputs(x, affine_offsets)


class AffineConv2d(nn.Conv2d, AffineKernal):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            last_affine=None      # 重用之前计算的映射值
    ):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(AffineConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )
        AffineKernal.__init__(self, kernel_size, stride)

        # self.stride = kernel_size   # 为了继承nn.Conv2d中的方法, 这里的stride是实际进行卷积时使用的stride
        # self.padding = 0
        # self.conv = super().forward

    def forward(self, x):
        x, affine_offsets = x
        kernal_x = self._get_conv_inputs(x, affine_offsets)
        # print('x:', kernal_x.shape, kernal_x.device)
        # print('w:', self.weight.shape, self.weight.device)
        # out = self.conv(kernal_x)
        out = super().forward(kernal_x)

        return out


class AffineMaxPool2d(nn.MaxPool2d, AffineKernal):

    def __init__(self, kernel_size=2, stride=None,
                 padding=0, dilation=1,
                 return_indices: bool = False, ceil_mode: bool = False):

        nn.MaxPool2d.__init__(self, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              return_indices=return_indices, ceil_mode=ceil_mode)
        AffineKernal.__init__(self, kernel_size, stride)

        # self.stride = kernel_size   # 为了继承nn.Conv2d中的方法, 这里的stride是实际进行卷积时使用的stride
        # self.padding = 0
        # self.pool = super().forward

    def forward(self, x):
        x, affine_offsets = x
        kernal_x = self._get_conv_inputs(x, affine_offsets)
        # out = self.pool(kernal_x)
        out = super().forward(kernal_x)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return AffineConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class AffineBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AffineBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x, affine_offsets = x
        residual = x

        out = self.conv1((x, affine_offsets))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2((out, affine_offsets))
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AffineBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AffineBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AffineConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.maxpool = AffineMaxPool2d(stride, stride, padding=0)

    def forward(self, x):
        x, affine_offsets = x
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2((out, affine_offsets))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # if self.stride == 1:
            #     residual = self.downsample(x)
            # else:
            #     residual = self.downsample(self.maxpool((x, affine_offsets)))
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        # return out
        return (out, affine_offsets)


class AffineResNet(nn.Module):
    def __init__(self, last_stride=2, block=AffineBottleneck, layers=[3, 4, 6, 3], pred_affine=True):
        self.inplanes = 64
        super().__init__()
        if pred_affine:
            self.affine_predictor = AffinePredictor()
        else:
            self.affine_predictor = nn.Sequential()
        self.affine_offsets = None
        self.conv1 = AffineConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = AffineMaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer0 = nn.Sequential(AffineConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #           kernel_size=1, stride=1, bias=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                # AffineMaxPool2d(kernel_size=stride, stride=stride, padding=0),      # 用 maxpool进行下采样
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, affine_predicts=None):
        if affine_predicts is None:
            affine_predicts = self.affine_predictor(x)
            if affine_predicts.dim() == 4:
                # affine_offset = affine_predicts[:, 0]
                affine_offset = affine_predicts.cumprod(dim=1)[:, -1]
                # affine_offset = cum_matmul(affine_predicts, dim=1)[:, -1]
            else:
                affine_offset = affine_predicts
        else:
            affine_offset = affine_predicts

        # affine_offset = torch.rand((x.shape[0], 2, 2), device=x.device)
        # print('a:', affine_offset.shape, affine_offset.device, )
        # print('inx:', x.shape, x.device)
        # print('ina:', affine_offset.shape, affine_offset.device)
        # print('inw:', self.conv1.weight.shape, self.conv1.weight.device)

        x = self.conv1((x, affine_offset))
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool((x, affine_offset))
        # x = self.layer0((x, affine_offset))      # downsample

        x, _ = self.layer1((x, affine_offset))
        x, _ = self.layer2((x, affine_offset))
        x, _ = self.layer3((x, affine_offset))
        x, _ = self.layer4((x, affine_offset))

        return x, affine_predicts

    # def get_affine(self):
    #     print(self.affine_offsets.shape)
    #     affine_offsets = self.affine_offsets
    #     self.affine_offsets = 1
    #     return affine_offsets

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        if 'model' in param_dict.keys():
            param_dict = param_dict['model']
        # if hasattr(param_dict, 'base'):
        #     param_dict = param_dict.base.state_dict()
        for i in param_dict:
            if 'fc' in i:
                continue
            if 'base' in i:
                j = i.replace('base.', '')
            else:
                j = i
            # self.state_dict()[j].copy_(param_dict[i])
            if j in self.state_dict().keys():
                self.state_dict()[j].copy_(param_dict[i])
            else:
                print('skip param ', j)
            # self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


