import numpy as np 
import cv2
import os
from os.path import join, split, exists


def mkdir(path):
    if not exists(path):
        os.makedirs(path)

data_split = 'sample'
dir_img = join('/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split, 'npy_img_dense')
dir_mask = join('/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split, 'npy_mask_6')
file = '0000/000004.npy'

img = np.load(join(dir_img, file))
print(img.shape)
img = np.transpose(img, (1, 2, 0)) * 255.
print(img.shape)

mask = np.load(join(dir_mask, file))
mask[np.where(mask == 1)] = 255
mask[np.where(mask == 0)] = 155
mask[np.where(mask == -1)] = 0

# img = np.expand_dims(img, axis=2)
img = np.concatenate((img, img, img), axis=-1)
mask = np.expand_dims(mask, axis=2)
mask = np.concatenate((mask, mask, mask), axis=-1)
print(mask.shape)

path = '0322GT'
mkdir(path)

# img = cv2.resize(img, (1920, 1216))
# mask = cv2.resize(mask, (1920, 1216))

cv2.imwrite(join(path, 'test_npy_img_or.png'), img)
cv2.imwrite(join(path, 'test_npy_mask_or.png'), mask)
