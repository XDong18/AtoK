import torch
from PIL import Image
import os
import cv2
from os import listdir
import logging
import numpy as np 
from torch.utils.data import Dataset
from glob import glob
from os.path import exists, join, split, dirname
import argparse
from skimage.draw import polygon
from density import get_neg_map, get_dense, get_dense_map


def parse_args():
    parser = argparse.ArgumentParser(
        description='generate npy files')
    parser.add_argument('data_split', choices=['train', 'val', 'test', 'sample'], 
                        help='Which data split to use in testing')
    parser.add_argument('-d','--dila', type=int, default=3)
    parser.add_argument('-m','--mask_name')
    args = parser.parse_args()

    return args

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = []
        files = listdir(imgs_dir)
        files.sort()
        for file in files:
            sub_img_dir = os.path.join(imgs_dir, file)
            image_files = listdir(sub_img_dir)
            image_files.sort()
            sub_ids = [os.path.join(file, image_f) for image_f in image_files]
            self.ids = self.ids + sub_ids
    
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_image(cls, img):
        args = parse_args()
        img = cv2.resize(img, (1920, 1216))
        # img[np.where(img==0)] = noise[np.where(img==0)]
        img = 255 - img
        img[np.where(img == 255)] = 0
        dila_point = np.logical_or.reduce(img > 0, axis=2)
        point_list = np.argwhere(dila_point==True)
        for point in point_list:
            cv2.circle(img, (point[1], point[0]), args.dila, 
            (int(img[point[0], point[1], 0]), int(img[point[0], point[1], 1]), int(img[point[0], point[1], 2])), -1)
        
        img_new = np.array([img[:, :, 0]])

        if img_new.max() > 1:
            img_new = img_new / 255

        return img_new
    
    @classmethod
    def preprocess_mask(cls, img, ignore_mask):
        args = parse_args()
        img = cv2.resize(img, (1920, 1216))
        # noise = noise * boxes
        # img[np.where(noise!=0)] = 1

        dila_point = np.logical_or.reduce(img > 0, axis=2)
        point_list = np.argwhere(dila_point==True)
        for point in point_list:
            cv2.circle(img, (point[1], point[0]), args.dila, 
            (int(img[point[0], point[1], 0]), int(img[point[0], point[1], 1]), int(img[point[0], point[1], 2])), -1)

        valid_point = np.logical_or.reduce(img > 0, axis=2)
        new_img = np.zeros((img.shape[0], img.shape[1]),dtype=np.float32)
        new_img[np.where(valid_point)] = 1
        new_img[np.where(ignore_mask)] = -1

        # dense_mask[np.where(new_img==1)] = 1
        # new_img[np.where(dense_mask==0)] = 0

        return new_img
    
    @classmethod
    def get_dense_mask(cls, img, the):
        img = img[:,:,0]
        H, xedges, yedges = get_dense(img, 1920/40, 1200/40)
        dense_map = get_dense_map(H, xedges, yedges)
        neg_map = get_neg_map(dense_map, the)
        return neg_map 
        
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx)
        img_file = glob(self.imgs_dir + idx)

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = cv2.imread(mask_file[0])
        img = cv2.imread(img_file[0])

        # img = cv2.resize(img, (1216, 1920))
        # mask = cv2.resize(mask, (1216, 1920))

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess_image(img)
        ignore_mask = img[0]==0
        mask = self.preprocess_mask(mask, ignore_mask)

        _, h, w = img.shape
        # img = img[:, int(h / 4):int(3 * h / 4), 0:int(3 * w/8)]
        # mask = mask[int(h / 4):int(3 * h / 4), 0:int(3 * w/8)]

        return(tuple([torch.from_numpy(img), torch.from_numpy(mask)]))
        # return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'ignore_mask': torch.from_numpy(ignore_mask)}

def mkdir(path):
    if not exists(path):
        os.makedirs(path)


if __name__=='__main__':
    args = parse_args()
    data_split = args.data_split

    # npy_img = join('/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split, 'npy_img_dense10')
    npy_mask = join('/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split, args.mask_name)

    # if not exists(npy_img):
    #     os.makedirs(npy_img)

    if not exists(npy_mask):
        os.makedirs(npy_mask)

    dir_img = '/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split + '/t_pc/'
    dir_mask = '/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split + '/instance_segment/'
    # dir_label = '/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split + '/label_02/'
    files = listdir(dir_img)
    files.sort()
    ids = []
    for file in files:
        sub_img_dir = os.path.join(dir_img, file)
        image_files = listdir(sub_img_dir)
        image_files.sort()
        sub_ids = [os.path.join(file, image_f.split('.')[0]) for image_f in image_files]
        ids = ids + sub_ids
    
    print("file created", len(ids))

    for i in range(len(ids)):
        # print(i)
        idx = ids[i]
        # print(dir_mask + idx)
        # print(dir_mask + idx)
        mask_file = glob(dir_mask + idx + '*')
        img_file = glob(dir_img + idx + '*')
        # label_file = glob(dir_label + idx + '*')
        # mkdir(join(npy_img, idx.split('/')[0]))
        mkdir(join(npy_mask, idx.split('/')[0]))
        mask = cv2.imread(mask_file[0])
        img = cv2.imread(img_file[0])
        # dense_mask = BasicDataset.get_dense_mask(img, 4)
        # boxes = BasicDataset.preprocess_bbox(label_file[0])
        # noise = BasicDataset.generate_noise(254, 0.003)
        img = BasicDataset.preprocess_image(img)
        ignore_mask = img[0]==0
        mask = BasicDataset.preprocess_mask(mask, ignore_mask)
        # np.save(join(npy_img, idx.split('.')[0] + '.npy'), img)
        np.save(join(npy_mask, idx.split('.')[0] + '.npy'), mask)
        if i%10 == 0:
            print("num", i)
        # if i==10:
        #     break






