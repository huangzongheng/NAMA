# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torch


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
        except OSError:
            print("OSError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, cache=False):
        self.dataset = dataset
        self.transform = transform
        if cache:
            self.cache = {}
        else:
            self.cache = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        # if img_path not in self.cache.keys():
        if self.cache is None:
            img = read_image(img_path)
        else:
            try:
                img = self.cache[img_path]
            except KeyError:
                img = read_image(img_path)
                self.cache[img_path] = img

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

class ImageDatasetEma(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, cache=False):
        self.dataset = dataset
        self.transform = transform
        if cache:
            self.cache = {}
        else:
            self.cache = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        # if img_path not in self.cache.keys():
        if self.cache is None:
            img = read_image(img_path)
        else:
            try:
                img = self.cache[img_path]
            except KeyError:
                img = read_image(img_path)
                self.cache[img_path] = img

        if self.transform is not None:
            # pre transform
            img = self.transform[2](img)

            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            img = torch.stack([img1, img2])

        return img, pid, camid, img_path
