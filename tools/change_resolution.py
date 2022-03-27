import cv2
import os
import shutil
import numpy as np
import random
import tqdm

random.seed(1024)
root= '/data/code/deep-metric-learning/datasets/DukeMTMC-reID/'
# source_dir = os.path.join(root, 'bounding_box_test')
# target_dir = os.path.join(root, 'hl_bounding_box_test')
source_dir = os.path.join(root, 'query')
target_dir = os.path.join(root, 'hl_query')

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)
for fname in tqdm.tqdm(os.listdir(source_dir)):
    os.symlink(os.path.join(source_dir, fname), os.path.join(target_dir, fname))
    img = cv2.imread(os.path.join(source_dir, fname))
    # scale = random.random() * 0.25 + 0.25
    # img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scale = int(random.random() * 10 + 30)
    img = cv2.resize(img, (scale, scale *2), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(target_dir, '_l.'.join(fname.split('.'))), img)
