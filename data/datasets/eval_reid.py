import torch
import numpy as np


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, return_AP=False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    # import pdb
    # pdb.set_trace()
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # indices = np.argsort(distmat, axis=1)
    indices = torch.argsort(distmat, dim=1)
    matches = (g_pids[indices] == q_pids[:, None]).int()

    AP_base = 1/torch.arange(1, num_g+1, dtype=torch.float)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # remove = (g_camids[order] == q_camid)               # filter all image under same camera
        # keep = np.invert(remove)
        keep = ~remove

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        # if not np.any(orig_cmc):
        if orig_cmc.sum() == 0:
            # this condition is true when query identity does not appear in gallery
            continue
        tmp_cmc = orig_cmc.cumsum(dim=0)
        cmc = (tmp_cmc[:max_rank] > 0).int()
        # cmc[cmc > 1] = 1

        all_cmc.append(cmc)
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        # tmp_cmc = orig_cmc.cumsum(dim=0)
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        # tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        tmp_cmc = orig_cmc * (tmp_cmc * AP_base[:len(tmp_cmc)])
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP.item())

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = torch.stack(all_cmc).numpy().astype(np.float32)
    # R1 = all_cmc[:, 0]
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    if return_AP:
        return all_cmc, mAP, all_AP     # R1

    return all_cmc, mAP


# import pyximport
# pyximport.install()
# try:
#     from .cy_eval_reid import *
# except ImportError as e:
#     pass

if __name__ == '__main__':
    import torch
    import math
    np.random.seed(0)
    nq, ng = 10000, 80000
    fdim = 2048
    q_pids = np.random.randint(0, 5, nq)
    g_pids = np.random.randint(0, 5, ng)
    q_camids = np.arange(nq)
    g_camids = np.arange(ng)
    qf = np.random.random((nq, fdim))
    gf = np.random.random((ng, fdim))
    q_pids, g_pids, q_camids, g_camids = map(torch.tensor, [q_pids, g_pids, q_camids, g_camids])
    import time
    # cdist
    start = time.time()
    gf = torch.tensor(gf)
    qf = torch.torch.tensor(qf)
    distmat = torch.cdist(qf, gf)# .numpy()
    # distmat = np.random.random((nq, ng))
    print(time.time()-start)
    # original
    # gf = torch.tensor(gf)
    # qf = torch.torch.tensor(qf)
    # start = time.time()
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(nq, ng) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(ng, nq).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.cpu().numpy()
    # print(time.time()-start)
    #
    # # split
    # start = time.time()
    # for i in range(math.ceil(nq/1000)):
    #     distmat[1000*i:1000*(i+1)] = torch.cdist(qf[1000*i:1000*(i+1)], gf)
    # print(time.time()-start)

    # import time
    start = time.time()
    print(eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, return_AP=True)[-1])
    print(time.time()-start)

