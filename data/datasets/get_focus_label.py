import os
import json
import torch
import numpy as np

# label_path = "/data/code/deep-metric-learning/datasets/DukeMTMC-reID/labels"
label_path = "/data/code/deep-metric-learning/datasets/DukeMTMC-reID/labels/hzh"
#
# focus=[]
# for fname in sorted(os.listdir(label_path)):
#     fpath = os.path.join(label_path, fname)
#     print(fname)
#     with open(fpath, 'r') as f:
#         label = json.load(f)
#         focus.append(torch.tensor(label['shapes'][0]['points'])//16)    # mean(1)
# focus = torch.stack(focus).int()[:, :, [1,0]]
# focus[:, 1] = (focus[:, 1]-focus[:, 0]).clamp_min(1)    # n, left top start(y,x), box_scale(y,x)
# print(focus.tolist())
def get_focus_label(a=None):
    files=[]
    focus=[]
    for fname in sorted(os.listdir(label_path)):
        fpath = os.path.join(label_path, fname)
        # print(fname)
        with open(fpath, 'r') as f:
            label = json.load(f)
            focus.append(torch.tensor(label['shapes'][0]['points'])/1)    # mean(1)
            files.append(label['imagePath'].split('\\')[-1])
    focus = torch.stack(focus)[:, :, [1,0]]
    focus[:, 1, 0].clamp_max_(16*(16-1))
    focus[:, 1, 1].clamp_max_(16*(8-1))
    focus[:, 1] = (focus[:, 1]-focus[:, 0]).clamp_min(16)
    return files, focus

#
# import time
# lasttime=time.time()
# for i in range(100):
#     # torch.randperm(100000)
#     np.random.choice(10000, 100000)
# print(time.time() - lasttime)