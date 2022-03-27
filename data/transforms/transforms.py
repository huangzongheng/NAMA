# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

# from __future__ import absolute_import

from torchvision.transforms import *
import torch
from PIL import Image
from collections.abc import Sequence, Iterable
import random
import math
import pdb
import numpy as np
import torchvision.transforms.functional as F
from math import ceil, floor, trunc

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                # if img.size()[0] == 3:
                if img.size()[0] >= 3:      # 不擦坐标
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[:, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    # img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomVerticalCropCont(object):
    def __init__(self, height,width):
        self.height = height
        self.width = width
    def __call__(self, img):
        w, h = img.size
        # ratio = min(1, np.random.uniform(0.7, 1.2))
        ratio = np.random.uniform(0.6, 1.1)
        ratio = float(ratio)#
        # jitter = np.random.uniform(0.9,1.11111)
        # apply_ratio = min(1.0, ratio*jitter)
        if ratio > 1.0:
            ratio += 0.1
        else:
            ratio -= 0.1
        apply_ratio = ratio
        start_ratio = 0
        if apply_ratio > 1.0:
            start_ratio = apply_ratio - 1
            apply_ratio = 2.0 - apply_ratio
        start_h = int(start_ratio*h)
        imgt = img.crop((0, start_h, w, np.round(h*apply_ratio)))
        imgt = imgt.resize((self.width, self.height), Image.BILINEAR)
        return [img, imgt, apply_ratio, start_ratio]


class RandomVerticalErase(object):
    def __init__(self, ratio, mode='pad'):
        self.ratio = ratio
        self.mode = mode
        self.rand = torch.distributions.Normal(0, 0.01)

    def __call__(self, img):
        # w, h = img.size
        # ratio = min(1, np.random.uniform(0.7, 1.2))
        if self.ratio < 1:
            ratio = np.random.uniform(self.ratio, 1.0)
            start_h = int(ratio * img.shape[1])
            # add mask to input
            mask = torch.ones(1, *img.shape[1:])

            if self.mode == 'pad':
                # img = torch.cat([img, mask], dim=0)
                img[:, start_h:].zero_()
            # elif self.mode == 'resize':   # 'resize'
            #     img = torch.nn.functional.interpolate(img[:, :start_h].unsqueeze(0),
            #                                           img.shape[1:], mode='bilinear', align_corners=False)[0]
            else:   # 随机选一种
                pass
            # affine_mat = torch.tensor([[1.,0,0,],
            #                            [0,ratio,0]])
            # affine_mat += self.rand.sample(affine_mat.shape)

        return img

class ListToTensor(object):
    def __init__(self, normalize):
        self.totensor = ToTensor()
        self.normalize = normalize
    def __call__(self,img_rp):
        # tensors = [self.totensor(i) for i in img_rp[:-1]]
        tensors = map(self.totensor, img_rp[:-2])
        tensors = map(self.normalize, tensors)
        return list(tensors) + [torch.tensor(i) for i in img_rp[-2:]]
        # return [*tensors, img_rp[-1]]

# resize and keep aspact ratio

class AutoResize(object):

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)

    def __init__(self, size, mode='', interpolation=Image.BILINEAR, factor=16):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.factor = factor
        self.interpolation = interpolation
        self.mode = mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size
        w = w*1.4
        ratio = min(*[i/j for i, j in zip(self.size, (h, w), )])  #

        exp_size = [round(i * ratio) for i in (h, w)]

        # F.center_crop()
        if self.mode == 'pad':
            img_r = F.resize(img, exp_size, self.interpolation)
            img_r = F.to_tensor(img_r)
            img_tmp = torch.zeros([3,] + self.size) + self.mean
            img_tmp[:, :img_r.shape[1], :img_r.shape[2]] = img_r
            return F.to_pil_image(img_tmp)
        elif self.mode == 'crop':
            final_size = [self.factor * round(i * ratio / self.factor) for i in (h, w)]     # img.size
            img = F.resize(img, exp_size, self.interpolation)
            img = F.pad(img, 10, (124, 116, 103))
            img = F.center_crop(img, final_size)
            return img
        else:
            return F.resize(img, self.size, self.interpolation)
            # return F.resize(img, self.size, self.interpolation)


        # return F.resize(img, self.size, self.interpolation)
        # return F.resize(img, final_size, self.interpolation)

class AddMask(object):
    def __init__(self, size, mode):
        self.size = size
        self.mode = mode

    def __call__(self, img):

        out = torch.zeros(4, *self.size)
        out[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        out[3, :img.shape[1], :img.shape[2]] = 1

        return out

def random_affine_img(img, xrange=(0.8,1.1), yrange=(0.5, 1.1), noise=0.05):
    n = img.shape[0]
    device = img.device
    affine = torch.zeros(n, 2, 3)

    # xrange = torch.tensor(xrange).log2()    # 对数刻度
    # yrange = torch.tensor(yrange).log2()
    xyrange = torch.tensor((xrange, yrange)).log2()

    scale = torch.rand(n, 2)*(xyrange[:, 1] - xyrange[:, 0]) + xyrange[:, 0]
    scale = torch.pow(2, scale)
    # shift = torch.rand(n, 2) * torch.tensor([2, 1]) - torch.tensor([1, 1.0])  # x(-1, 1), y(-1, 0)
    # shift = shift * ((1 - scale).clamp_min(0) + (scale + 1).clamp_max(0))
    abs_scale = scale.abs()
    shift = (torch.rand(n, 2) - 0.5)*2
    shift = shift * (torch.max(abs_scale, 1/abs_scale) - (1 - 0.2))

    # affine[:, 0, 0] = torch.rand(n)*(xrange[1] - xrange[0]) + xrange[0]   # ratio x
    # affine[:, 1, 1] = torch.rand(n)*(yrange[1] - yrange[0]) + yrange[0]   # ratio_y
    # affine[:, 0, 0] = torch.pow(2, affine[:, 0, 0])
    # affine[:, 1, 1] = torch.pow(2, affine[:, 1, 1])
    # affine[:, 1, 2] = (affine[:, 1, 1] - 1)      # 保证只crop下半身

    affine[:, 0, 0] = scale[:,0]
    affine[:, 1, 1] = scale[:,1]
    affine[:, 0, 2] = shift[:,0]
    affine[:, 1, 2] = shift[:,1]
    # affine[:, 1, 2] = (affine[:, 1, 1] - 1)      # 保证只crop下半身
    affine += torch.rand(n, 2, 3) * noise - noise/2
    affine = affine.to(device)

    grid = torch.nn.functional.affine_grid(affine, img.shape, align_corners=False)
    img_affine = torch.nn.functional.grid_sample(img, grid, padding_mode='zeros')  # reflection, zeros

    return img_affine, affine


class DownCropResize(object):
    def __init__(self, ratio=1.0):
        self.ratio = ratio

    def __call__(self, img):

        out = torch.nn.functional.interpolate(img[:, :int(img.shape[1] * self.ratio)].unsqueeze(0),
                                              img.shape[1:], mode='bilinear')
        # out = torch.nn.functional.interpolate(img[:, :, :int(img.shape[2] * self.ratio)].unsqueeze(0),
        #                                       img.shape[1:], mode='bilinear')
        return out.squeeze()

# return (R, G, B, Hind, Wind), H,W
class AddPosMask(object):
    def __init__(self, start=0):
        self.start = start

    def __call__(self, img):
        h = torch.arange(img.shape[-2])[:, None].expand(*img.shape[:-3], 1, *img.shape[-2:]) + self.start
        w = torch.arange(img.shape[-1])[None, :].expand(*img.shape[:-3], 1, *img.shape[-2:]) + self.start
        img = torch.cat([img, h, w], dim=-3)

        return img

class ResumePosMask(object):
    def __init__(self, start=0):
        self.start = start

    def __call__(self, img):
        img[..., 3:,:,:] -= self.start

        return img

