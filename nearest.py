import numpy as np
import os 
from os.path import join, split, exists
import cv2
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from glob import glob
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='generate npy files')
    parser.add_argument('data_split', choices=['train', 'val', 'test', 'sample'], 
                        help='Which data split to use in testing')
    parser.add_argument('-j', '--jobs', default=1, type=int)
    args = parser.parse_args()

    return args


def mkdir(path):
    if not exists(path):
        os.makedirs(path)

def knn(mask):
    # print(mask)
    args = parse_args()
    positive_list = np.argwhere(mask==1)
    positive_label = np.ones((len(positive_list)))
    # print(positive_list)

    negative_list = np.argwhere(mask==0)
    negative_label = np.zeros((len(negative_list)))
    # print(negative_list)

    query_list = np.concatenate([positive_list, negative_list], axis=0)
    query_label = np.concatenate([positive_label, negative_label], axis=0)

    ignore_list = np.argwhere(mask==-1)
    # print("time: %.2f"%(time.time()-end))

    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=args.jobs)
    # print("time: %.2f"%(time.time()-end))

    neigh.fit(query_list, query_label)
    # print("time: %.2f"%(time.time()-end))

    ignore_label = neigh.predict(ignore_list)
    # print("time: %.2f"%(time.time()-end))

    new_mask = mask
    new_mask[np.where(mask==-1)] = ignore_label

    return new_mask

if __name__=='__main__':
    args = parse_args()
    data_split = args.data_split
    npy_mask_target = join('/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split, 'npy_mask_knn')

    if not exists(npy_mask_target):
        os.makedirs(npy_mask_target)
    
    dir_mask = '/shared/xudongliu/data/argoverse-tracking/argo_track/' + data_split + '/npy_mask_dense/'
    files = listdir(dir_mask)
    files.sort()
    ids = []
    for file in files:
        sub_img_dir = os.path.join(dir_mask, file)
        image_files = listdir(sub_img_dir)
        image_files.sort()
        sub_ids = [os.path.join(file, image_f.split('.')[0]) for image_f in image_files]
        ids = ids + sub_ids
    
    print("file created", len(ids))

    for i in range(len(ids)):
        end = time.time()
        # print(i)
        idx = ids[i]
        # print(dir_mask + idx)
        # print(dir_mask + idx)
        mask_file = glob(dir_mask + idx + '*')
        # img_file = glob(dir_img + idx + '*')
        # label_file = glob(dir_label + idx + '*')
        # mkdir(join(npy_img, idx.split('/')[0]))
        mkdir(join(npy_mask_target, idx.split('/')[0]))
        mask = np.load(mask_file[0])
        # img = cv2.imread(img_file[0])
        # dense_mask = BasicDataset.get_dense_mask(img, 4)
        # boxes = BasicDataset.preprocess_bbox(label_file[0])
        # noise = BasicDataset.generate_noise(254, 0.003)
        # img = BasicDataset.preprocess_image(img)
        # ignore_mask = img[0]==0
        # mask = BasicDataset.preprocess_mask(mask, ignore_mask, dense_mask)
        mask = knn(mask)
        # np.save(join(npy_img, idx.split('.')[0] + '.npy'), img)
        np.save(join(npy_mask_target, idx.split('.')[0] + '.npy'), mask)
        if i%5 == 0:
            print("num", i, 'time: %.2f0'%(time.time()-end))
            end = time.time()
        # if i==5:
        #     break




