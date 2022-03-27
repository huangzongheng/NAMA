import torch
import functools

def cum_matmul(mats : torch.Tensor, dim=0):
    assert mats.shape[-1] == mats.shape[-2]
    out = []
    for mat in mats.unbind(dim):
        if len(out) > 0:
            out.append(torch.matmul(out[-1], mat))
        else:
            out.append(mat)
    return torch.stack(out, dim=dim)


class GussianBlur(torch.nn.Module):
    def __init__(self, kernal_size=3, std=1):
        super(GussianBlur, self).__init__()
        grid = (torch.arange(kernal_size).float() - (kernal_size-1)/2)/std
        x = grid.reshape(1, -1)
        y = grid.reshape(-1, 1)
        kernal = -x.pow(2) - y.pow(2)
        kernal = kernal.exp()

        self.register_buffer('kernal', kernal/kernal.sum())
        self.conv = functools.partial(torch.nn.functional.conv2d,
                                      padding=(kernal_size-1)//2)

    def forward(self, x, *args, **kwargs):
        channels = x.shape[-3]
        return self.conv(x, self.kernal.expand(channels, 1, *self.kernal.shape), groups=channels)

