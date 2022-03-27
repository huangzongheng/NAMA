# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
from .affine_conv import AffineConv2d

last_affine=None

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BatchNormPadding2d(nn.BatchNorm2d):


    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, padding=0):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input, mask=None):
        # 将非mask部分的均值方差与mask部分对齐，防止影响bn统计量计算
        if self.training and mask is not None:
            masked_input = mask * input
            mask_rate = self.gap(mask)
            masked_mean = self.gap(masked_input) / mask_rate    # 有效图像均值
            pad = (1-mask) * masked_mean    # pad对齐有效图像均值
            masked_std = (masked_input + pad).reshape(*masked_input.shape[:2], -1).std(-1)  # 有效图像标准差
            masked_std = masked_std.unsqueeze(-1).unsqueeze(-1) / torch.sqrt(mask_rate.clamp_min(1e-4))
            pad[..., :pad.shape[-1]//2] += masked_std       # 左加右减，使标准差对齐
            pad[..., pad.shape[-1]//2:] -= masked_std
            input = masked_input + (1-mask) * pad.detach()

        out = super().forward(input)    # 标准BN计算
        # bn和affine之间加padding
        if self.padding > 0:
            b = self.bias.view(-1, 1, 1)
            out = out - b
            if mask is not None:
                out = out * mask
            out = self.pad(out)
            out = out + b

        return out, mask


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv=nn.Conv2d):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        global last_affine
        if conv == AffineConv2d:
            self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, last_affine=last_affine)
        else:
            self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(stride, stride, 0)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # if self.stride == 1:
            #     residual = self.downsample(x)
            # else:
            #     residual = self.downsample(self.maxpool(x))
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckHomo(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckHomo, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNormPadding2d(planes, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=0, bias=False)
        self.bn2 = BatchNormPadding2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNormPadding2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(x, tuple):
            x, mask = x
        else:
            mask=None
        identity = x

        out = self.conv1(x)
        out, mask = self.bn1(out, mask)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out, _ = self.bn3(out)

        if self.downsample is not None:
            identity, _ = self.downsample(x)
            if mask is not None:
                mask = nn.functional.avg_pool2d(mask, self.stride)

        out += identity
        out = self.relu(out)

        return out, mask


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], affine=False):
        self.inplanes = 64
        super().__init__()
        if block == BottleneckHomo:
            norm = BatchNormPadding2d
        else:
            norm = nn.BatchNorm2d
        if affine is True:
            self.conv_type = AffineConv2d
        else:
            self.conv_type = nn.Conv2d
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = self.conv_type(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        global last_affine
        last_affine = self.conv1
        self.bn1 = norm(64)   # nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer0 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if block == BottleneckHomo:
            norm = BatchNormPadding2d
        else:
            norm = nn.BatchNorm2d
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #           kernel_size=1, stride=1, bias=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm(planes * block.expansion),
                # nn.BatchNorm2d(planes * block.expansion),
                # nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0),      # 用 maxpool进行下采样
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.conv_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_pyramid=False):
        if x.shape[1] == 4:
            mask = x[:, -1:]
            mask = nn.functional.avg_pool2d(mask, 2)
            x = x[:, :-1]
        else:
            mask = None

        if isinstance(self.bn1, BatchNormPadding2d):
            x = self.conv1(x)
            x, mask = self.bn1(x, mask)
            # x = self.relu(x)
            x = self.maxpool(x)
            if mask is not None:
                mask = nn.functional.avg_pool2d(mask, 2)

            x, mask = self.layer1((x, mask))
            x, mask = self.layer2((x, mask))
            x, mask = self.layer3((x, mask))
            x, mask = self.layer4((x, mask))
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)    # add missed relu
            x = self.maxpool(x)
            # x = self.layer0(x)

            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x = self.layer4(x3)
        if return_pyramid:  # 返回中层特征
            x = x, (x1, x2, x3, x)
        return x

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

    def get_affine(self):
        global last_affine
        if isinstance(last_affine, AffineConv2d):
            return last_affine.cur_affine
        else:
            return None

