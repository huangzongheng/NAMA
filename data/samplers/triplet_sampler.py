# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, seed=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.rng = np.random.default_rng(seed)

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = self.rng.choice(idxs, size=self.num_instances, replace=True)
                # idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            # random.shuffle(idxs)
            self.rng.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        remains = np.array([len(batch_idxs_dict[i]) for i in range(len(avai_pids))])
        # while len(avai_pids) >= self.num_pids_per_batch:
        while len(remains.nonzero()[0]) >= self.num_pids_per_batch:
            # selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            # id采样概率与其剩余样本数成正比  .clip(0, 1)
            selected_pids = self.rng.choice(len(remains), self.num_pids_per_batch, replace=False,
                                            p=(remains/remains.sum()))
            remains[selected_pids] -= 1
            # print(remains[463], remains[463]/remains.sum())
            # if sum(remains[selected_pids] < 0) >0:
            #     print(selected_pids)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                # if len(batch_idxs_dict[pid]) == 0:
                #     avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

        # def final_idxs():
        #     while len(avai_pids) >= self.num_pids_per_batch:
        #         selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
        #         for pid in selected_pids:
        #             batch_idxs = batch_idxs_dict[pid].pop(0)
        #             # final_idxs.extend(batch_idxs)
        #             yield batch_idxs
        #             if len(batch_idxs_dict[pid]) == 0:
        #                 avai_pids.remove(pid)
        # final_idxs = iter(final_idxs())
        # self.length = len(list(final_idxs))
        # return final_idxs

    def __len__(self):
        return self.length

# Graph Sampling
class GlobalHardMiningSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, K=4, feature_dim=2048):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.K = K
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.bank = torch.rand(len(self.pids), feature_dim)
        torch.nn.init.normal_(self.bank, mean=0, std=1e-4)

        # estimate number of examples in an epoch
        self.length = self.pids * self.num_instances * self.K

    def get_topk(self):
        dist_mat = torch.cosine_similarity(self.bank[:, None], self.bank, dim=-1)
        batch_cind = dist_mat.topk(self.K, dim=-1)[1]
        return batch_cind

    def __iter__(self):
        # batch_idxs_dict = defaultdict(list)
        batch_cinds = self.get_topk()
        final_idxs = []
        # for cinds in batch_cinds:
        #     for pid in cinds:
        for sub_batch_id in torch.randperm(batch_cinds.shape[0]):
            for pid in batch_cinds[sub_batch_id]:
                idxs = self.index_dic[pid.item()]
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                else:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=False)
                # random.shuffle(idxs)
                final_idxs.extend(idxs)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
