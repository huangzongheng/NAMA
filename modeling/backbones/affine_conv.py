import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Reduce, Rearrange
from torchvision.models import resnet18


class AffineConv2d(nn.Conv2d):

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

        self.kernel_size = kernel_size
        self.padding = 0
        self.real_stride = stride   # 与正常卷积等效的stride
        self.cur_affine = None      # 本次forward 计算的affine mat

        self.zero_padding = nn.ZeroPad2d(padding)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.stride = kernel_size   # 为了继承nn.Conv2d中的方法, 这里的stride是实际进行卷积时使用的stride
        self.conv = super().forward
        # self.kernel_size_2 = self.kernel_size**2
        # self.conv = nn.Conv2d(inc, outc*self.kernel_size, kernel_size=1, stride=1, bias=bias, groups=self.kernel_size)
        if isinstance(last_affine, AffineConv2d):
            self.last_affine = last_affine
            self.get_affine = None
        else:
            self.last_affine = None
            cnn = resnet18(pretrained=True)
            self.get_affine = nn.Sequential(
                cnn.conv1,
                cnn.bn1,
                cnn.relu,
                cnn.maxpool,
                cnn.layer1,     # 64
                # cnn.layer2,     # 128
                # cnn.layer3,     # 256
                # nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, stride=1),
                # nn.BatchNorm2d(128),
                # nn.ReLU(),
                # nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.AdaptiveAvgPool2d((1,1)),
                # nn.Conv2d(64, 6, kernel_size=3, padding=1, stride=1),   # affine mat
                # nn.Conv2d(64, 4, kernel_size=3, padding=1, stride=1),   # delta x1 y1, x2 y2
                # nn.Flatten(),
                Reduce('b c h w -> b c', 'mean'),
                nn.Linear(64, 256),
                nn.ReLU(True),
                nn.Linear(256, 4),

                Rearrange('b (x y) -> b x y', x=2, y=2)     # y=3
                                            )
            nn.init.constant_(self.get_affine[-2].weight, 0)
            self.get_affine[-2].bias.data = torch.tensor([1.0, 0., 0., 1.0])  # x1, y1, x2, y2
            self.get_affine.register_backward_hook(self._set_lr)
            # self.get_affine[-3].bias.data = torch.tensor([1.5, 0., 0., 0., 1.5, 0.])  # x scale； y scale
            # 定义上边和右面的像素坐标，其余pixel以此为基础计算delta
        # nn.init.constant_(self.get_affine[0].bias, 0)
        # nn.init.normal_(self.get_affine[0].weight, 0, 1e-5)

        # nn.init.normal_(self.get_affine[0].bias, 0, 1e-4)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def _get_conv_inputs(self, x, affine_offset):   # affine: n, 2, 2(x, y)
        n, c, h, w = x.shape
        out = []
        grid_base = F.affine_grid(theta=torch.tensor([[1.0, 0., 0.], [0., 1.0, 0.]], device=x.device).expand(1, 2, 3),
                                  size=[1, c, h, w])    # n, h, w, 2(x,y)
        grid_base = grid_base[:, ::self.real_stride, ::self.real_stride]

        scale = 2 / torch.tensor([w, h], dtype=torch.float, device=x.device)   # convert pixel coordinate to [-1, 1] scale
        offset_base = torch.arange(self.kernel_size, device=x.device).reshape(-1, 1, 1, 1) - self.kernel_size//2   # [-1, 0, 1]...
        offset_base = offset_base * affine_offset   # k, n, 2, 2(x,y)  a*[-1, 0, 1]*scale
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
            grid = scale * (offset_base[j, :, 1] + offset_base[i, :, 0])
            grid = rearrange(grid, "n b-> n 1 1 b", b=2) + grid_base
            out.append(F.grid_sample(x, grid))

        # base_offsets = torch.meshgrid(
        #     [torch.arange(self.kernel_size),
        #      torch.arange(self.kernel_size)])
        # x = self.zero_padding(x)
        # generate affine grid for every pixel of conv layer
        # for oy, ox in zip(*[i.flatten().tolist() for i in offsets]):
        #     out.append(F.grid_sample(x[..., oy:oy+h:self.real_stride, ox:ox+w:self.real_stride], grid))
            # out.append(x[..., oy:oy+h:self.real_stride, ox:ox+w:self.real_stride])
        # out = torch.stack(out, dim=1)     # n*(9*c)*h*w
        out = rearrange(out, '(k k1) b c h w -> b c (h k) (w k1)', k=self.kernel_size, k1=self.kernel_size)

        return out

    def forward(self, x):
        n, c, h ,w = x.shape
        if self.last_affine is None:
            affine_mat = self.get_affine(x)
        else:
            affine_mat = self.last_affine.cur_affine
        self.cur_affine = affine_mat        # 记录当前变换值，给后面复用
        # grid = F.affine_grid(theta=affine_mat, size=(n, c, h//self.real_stride, w//self.real_stride))
        kernal_x = self._get_conv_inputs(x, affine_mat)
        out = self.conv(kernal_x)

        return out

