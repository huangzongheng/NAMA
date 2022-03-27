from __future__ import absolute_import
from __future__ import print_function

__all__ = ['visualize_ranked_results']

import numpy as np
import torch
import os
import os.path as osp
import shutil
import cv2
import math
import pdb
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')

from .iotools import mkdir_if_missing


GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
BLUE = (255, 0, 0)
YELLOW = (0, 192, 192)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
FRONT = cv2.FONT_HERSHEY_SIMPLEX
DOWNSAMPLE=1

def visualize_ranked_results(distmat, dataset, data_type='image', width=128, height=256, save_dir='', topk=10,
                             attr=None, focus=None, distmap=None, AP=None, norms=None):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    # query, gallery = dataset
    query = dataset[:num_q]
    gallery = dataset[num_q:]
    gpids = np.array([i[1] for i in gallery])
    gcids = np.array([i[2] for i in gallery])
    # assert num_q == len(query)
    # assert num_g == len(gallery)
    assert num_g + num_q == len(dataset)

    indices = np.argsort(distmat, axis=1)
    if attr is not None:
        attr = attr > 0

    if AP is not None:
        assert len(AP) == num_q

    if norms is not None:
        assert len(norms) >= num_q
        q_norms = norms[:num_q]
        g_norms = norms[num_q:]


    # DOWNSAMPLE = downsample
    if focus is not None:
        qfocus, gfocus = focus
        qfocus = qfocus * DOWNSAMPLE
        gfocus = gfocus * DOWNSAMPLE

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)


    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            # qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            # grid_img = 255 * np.ones((height, num_cols*width+topk*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
            grid_img = 255 * np.ones((math.ceil((topk/10)) * height + max(0, (topk-1)//10) * GRID_SPACING,
                                      11*width+10*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
            grid_img[:height, :width, :] = qimg
            if attr is not None:
                # grid_img = cv2.putText(grid_img, str(qpid), (width//2 - 20, height - 10), FRONT,
                #                        0.6, RED, 2)
                grid_img = print_text_with_color(grid_img, 0, height - 10, attr[q_idx],)
            else:   # query id
                grid_img = cv2.putText(grid_img, str(qpid), (width//2 - 20, height - 10), FRONT,
                                       0.6, RED, 2)
            if focus is not None:       # highlight query focus
                grid_img = cv2.rectangle(grid_img, tuple(qfocus[q_idx,0].int().tolist()[::-1]),
                                         tuple((qfocus[q_idx].int().sum(0)).tolist()[::-1]), GREEN, thickness=2)

            if AP is not None:
                grid_img = cv2.putText(grid_img, "{:.2f}".format(AP[q_idx]), (width+2, 25), FRONT,
                                       0.7, RED, 2)

            if norms is not None:
                grid_img = cv2.putText(grid_img, "{:.1f}".format(q_norms[q_idx]), (width+2, 50), FRONT,
                                       0.7, YELLOW, 2)
        else:
            qdir = osp.join(save_dir, osp.basename(osp.splitext(qimg_path_name)[0]))
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx,:]:

            if rank_idx >= topk:        # 最后一张图显示排在最后面的TP，不代表rank
                # pdb.set_trace()
                valid = (qpid == gpids[indices[q_idx,:]]) & (qcamid != gcids[indices[q_idx,:]])
                # g_idx = indices[q_idx, valid][-1]
                g_idx = indices[q_idx, np.nonzero(valid)[-1][-1]]

            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            # invalid = (qcamid == gcamid)

            if not invalid:
                matched = gpid==qpid
                if data_type == 'image':
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))

                    if distmap is not None:
                        # activation map
                        am = distmap[q_idx, g_idx, ...].numpy()
                        am = cv2.resize(am, (width, height))
                        am = 255 * (-am + np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
                        am = np.uint8(np.floor(am))
                        am = cv2.applyColorMap(am, cv2.COLORMAP_JET) * (am[..., None] > 100)
                        # cv2.findContours

                        # overlapped
                        overlapped = gimg * 1.0 + am * 0.2
                        np.clip(overlapped, 0, 255, out=overlapped)
                        gimg = overlapped.astype(np.uint8)

                    border_color = GREEN if matched else RED
                    gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    gimg = cv2.resize(gimg, (width, height))
                    start = ((rank_idx - 1) % 10 + 1)*(width + GRID_SPACING) + QUERY_EXTRA_SPACING
                    end = start + width
                    hstart = (math.ceil(rank_idx/10)-1) * (height + GRID_SPACING)
                    hend = hstart + height

                    try:
                        grid_img[hstart:hend, start: end, :] = gimg
                        assert start < 2000
                        assert hstart < 2000
                    except:
                        pdb.set_trace()
                    # grid_img[:, start: end, :] = gimg

                    # similarity
                    # grid_img = cv2.putText(grid_img, "{:.1f}".format((2-distmat[q_idx, g_idx])*50),
                    #                        (start+5, hstart + 25), FRONT, 0.7, GREEN, 2)

                    if norms is not None:
                        grid_img = cv2.putText(grid_img, "{:.1f}".format(g_norms[g_idx]),
                                               (start+70, hstart + 25), FRONT, 0.7, YELLOW, 2)
                    if attr is not None:
                        grid_img = print_text_with_color(grid_img, start, hend - 5, attr[g_idx+num_q],
                                                            attr[q_idx].eq(attr[g_idx+num_q]))
                    else:   # gallery id
                        # filename
                        # grid_img = cv2.putText(grid_img, gimg_path.split('/')[-1].split('.')[0],
                        #                        (start, hend - 5), FRONT, 0.5, RED, 2)
                        pass
                    # grid_img = cv2.putText(grid_img, str(gpid), (start + width//2 - 20, height - 10), front,
                    #                        0.6, RED, 2)

                    if focus is not None:       # highlight gallery matched focus
                        gf = gfocus[q_idx, g_idx]
                        grid_img = cv2.rectangle(grid_img, (gf[1]+start, gf[0]+hstart, ),
                                                 (gf[1]+qfocus[q_idx,1,1].int()//DOWNSAMPLE*DOWNSAMPLE+start,
                                                  gf[0]+qfocus[q_idx,1,0].int()//DOWNSAMPLE*DOWNSAMPLE+hstart),
                                                 GREEN, thickness=2)
                else:
                    _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery', matched=matched)

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            if norms is not None:
                norm_folder = str(int(norms[q_idx].item()/2)*2)
                mkdir_if_missing(osp.join(save_dir, norm_folder))
                imname = osp.join(norm_folder, imname)
            # norm_folder = str(int(10*AP[q_idx]))
            # mkdir_if_missing(osp.join(save_dir, norm_folder))
            # imname = osp.join(norm_folder, imname)
            cv2.imwrite(osp.join(save_dir, imname+'.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx+1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))

def print_text_with_color(img, width, height, attr, target=None, color=(RED, GREEN)):
    if target is None:
        target = [1]*len(attr)
    for i, a in enumerate(attr):
        img = cv2.putText(img, str(a.int().item()), (width + (8*i)%128, height+12*((i//16)-1)), FRONT,
                          0.5, color[target[i]], 2)
    return img

def vis_AP(APs: list, save_dir='./'):
    # APs.sort(reverse=True)
    bins = np.arange(0, 1.1, 0.05)
    plt.hist(APs, bins)
    plt.xlabel('imgs')
    plt.ylabel('AP')
    plt.title('AP for each images')
    plt.savefig(os.path.join(save_dir, 'APs.jpg'))


def vis_norm_uc(norms, uc, bins=50, save_dir='./'):
    # 计算归一化模长
    org_norms = norms
    # m_norm = norms.mean()
    # n_norm = torch.log(norms / m_norm)
    for norms, fname in zip([org_norms, org_norms.log()],
                      ['norm_uc.png', 'log_norm_uc.png']):
        m_norm = norms.mean()
        if fname == 'norm_uc.png':
            # n_norm = (norms - m_norm) / m_norm
            n_norm = norms
        else:
            n_norm = (norms - m_norm)
        # 初始化bins
        n_min = n_norm.min()
        n_max = n_norm.max()
        i_norm = (n_norm-n_min)/(n_max-n_min)
        i_norm = (bins * (i_norm) + 0.5).int()
        counts = torch.bincount(i_norm)/float(norms.shape[0])
        cx = torch.linspace(n_min, n_max, bins+1)
        fig = plt.figure()
        if len(uc) == len(norms):
            # plt.scatter(n_norm.numpy(), uc.numpy(), label='uc', s=1, c='#ff0000', marker='.', alpha=0.1)
            plt.scatter(n_norm.numpy(), -2*uc.numpy(), label='margin', s=1, c='#ff0000', marker='.', alpha=0.1)

        plt.legend()
        plt.xlabel('norms')
        # plt.ylabel()
        plt.show()
        plt.savefig(os.path.join(save_dir, fname.replace('norm_uc', 'margin')))
        plt.plot(cx.numpy(), (2*counts-counts.max()).numpy(), label='norm', c='#00bb00')
        plt.savefig(os.path.join(save_dir, fname))

# fig = plt.figure()
# plt.plot(cx.numpy(), ((counts/counts.max())*(uc.max()-uc.min()+0.025)+uc.min()).numpy()-0.05, label='density', c='#00bb00')
# plt.scatter(n_norm.numpy(), uc.numpy(), s=1, c='#ff0000', alpha=0.1)
# plt.scatter(n_norm[:1].numpy(), uc[:1].numpy(), label='margin', s=1, c='#ff0000', alpha=1)
# plt.legend()
# plt.xlabel('norms')
# plt.ylabel('margins')
# plt.savefig('./norm.png', dpi=300)
#
# import time
# fig = plt.figure()
# for i in (2, 8, 32, 128):
#     data = torch.load('/root/code/dml/logs/spl/cls+/market1501/Exp-log-nn_{:d}-d1e-2-bnneck-uc-cls-arc-10*12-s16-m0.0-g0.0/norm.pt'.format(i))
#     ind = data['norm'].argsort()
#     data['norm'][ind[0]] = 20
#     data['norm'][ind[-1]] = 70
#     plt.plot(data['norm'][ind].numpy().clip(20, 70), -2*data['uc'][ind].numpy(), label='dh={:d}'.format(i))
#     time.sleep(1)
#     print(i, 'complete')
# plt.legend()
# plt.savefig('dim.png')

if __name__ == '__main__':
    import torch
    import time
    from modeling.uc_base import Norm2UC
    class M(torch.nn.Module):
        def __init__(self, ndim):
            super(M, self).__init__()
            self.bn_norm = torch.nn.BatchNorm1d(1)
            self.norm2uc = Norm2UC(n_dim=ndim)
            self.register_buffer('avg_uc', torch.tensor(0.0))

        def forward(self, x):
            x = self.bn_norm(x[:, None])[..., 0]
            return self.norm2uc(x) - self.avg_uc

    fig = plt.figure()
    data = {}
    inputs = torch.linspace(20, 70, 100)
    data[0] = inputs.detach().numpy().tolist()
    for i in (2, 8, 32, 128, 512):
        ckp = torch.load(
            '/root/code/dml/logs/spl/cls+/market1501/Exp-log-nn_{:d}-d1e-2-bnneck-uc-cls-arc-10*12-s16-m0.0-g0.0/resnet50_checkpoint_180.pt'.format(i),
            map_location='cpu')
        model = M(i)
        model.eval()
        model.load_state_dict(ckp['model'], strict=False)
        out = model(inputs)
        plt.plot(inputs.numpy().clip(20, 70), -2 * out.detach().numpy(), label='dh={:d}-'.format(i))
        time.sleep(1)
        print(i, 'complete')
        data[i] = (-2 * out.detach().numpy()).tolist()
    plt.legend()
    plt.savefig('dim.png', dpi=300)


    import time
    fig = plt.figure()
    for i in (2, 8, 32, 128, 512):
        data = torch.load('/root/code/dml/logs/spl/cls+/market1501/Exp-log-nn_{:d}-d1e-2-bnneck-uc-cls-arc-10*12-s16-m0.0-g0.0/norm.pt'.format(i))
        ind = data['norm'].argsort()
        data['norm'][ind[0]] = 20
        data['norm'][ind[-1]] = 70
        plt.plot(data['norm'][ind].numpy().clip(20, 70), -2*data['uc'][ind].numpy(), label='dh={:d}'.format(i))
        time.sleep(1)
        print(i, 'complete')
    plt.legend()
    plt.savefig('dim.png')
