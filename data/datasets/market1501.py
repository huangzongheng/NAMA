# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os
import mat4py
import numpy as np

import os.path as osp

from .bases import BaseImageDataset


MARKET_ATTR_PARTS = {
    'head':['hair',
            'hat',],
    'up':['upblack',
          'upblue',
          'upgreen',
          'upgray',
          'uppurple',
          'upred',
          'upwhite',
          'upyellow',],
    'arm':['up',],
    'down':['downblack',
            'downblue',
            'downbrown',
            'downgray',
            'downgreen',
            'downpink',
            'downpurple',
            'downwhite',
            'downyellow',],
    'leg':['down','clothes',],
    # 'shoe':[],
    'bag':['backpack', 'bag', 'handbag',],
    'global':['age',
              'gender',]
}

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root='/home/haoluo/data', verbose=True, use_attr=False, combine_all=False, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        if combine_all:
            train = self._process_dir([self.train_dir, self.query_dir, self.gallery_dir], relabel=True)
        else:
            train = self._process_dir(self.train_dir, relabel=True)

        # train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        # self.train = train[:64]
        # self.query = query[-16:]
        # self.gallery = gallery[-64:]

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        if use_attr:
            attrs = mat4py.loadmat(os.path.join(self.dataset_dir,'attribute','market_attribute.mat'))['market_attribute']
            attr_train = attrs['train']
            attr_test = attrs['test']
            for age in (1, 2, 3, 4):    # 将年龄展开为4个二分类问题
                attr_train['age'+str(age)] = [2 if i == age else 1 for i in attr_train['age']]
            attr_train.pop('age')
            # attr_train['age0'] = [2 if i == age else 1 for i in attr_train['age']]


            cls = set(attr_train.keys())
            cls.discard('image_index')
            cls = list(cls)
            cls.sort()

            c = np.zeros((len(attr_train['image_index']), len(cls)+1))    # +1 for id label
            for i in range(len(cls)):
                c[:, i] = attr_train[cls[i]]
            c[:, -1] = np.arange(c.shape[0]) + 1
            self.train_attr = c.astype(np.int64) - 1
            self.classes_attr = cls
            self.train = [[i[0], i[1], self.train_attr[i[1]]] for i in self.train]


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        if isinstance(dir_path, str):
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths=[]
            for d in dir_path:
                img_paths += glob.glob(osp.join(d, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                if pid < 0:
                    continue
                pid = pid2label[pid]
            # if relabel: pid = self.train_attr[pid]      # use attr & id label
            dataset.append((img_path, pid, camid))

        return dataset


class Market1501Partial(Market1501):
    dataset_dir = 'Market-1501-Partial-Head'
    # dataset_dir = 'Market-1501-Partial'

