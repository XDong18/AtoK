import numpy as np 
import os
from os.path import join, split
import argparse
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt


def get_dense(img, bin_x, bin_y):
    '''注意h,w和x,y的变换。在本函数中，所有x,y都是正常的x和y.'''
    points_list = np.argwhere(img>0)
    ys = points_list[:,0]
    xs = points_list[:,1]
    H, xedges, yedges = np.histogram2d(xs, ys, bins=[bin_x, bin_y])
    return H, xedges, yedges

def get_dense_map_spline(H, xedges, yedges, shape=(1200, 1920)):
    h, w = shape
    # neg_map = np.zeros(shape)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xx, yy = np.meshgrid(xcenters, ycenters)
    H = H.T 
    # f = interpolate.interp2d(xcenters, ycenters, H)
    tck = interpolate.bisplrep(xx, yy, H, s=0)
    xnew = np.arange(0, w, 1)
    ynew = np.arange(0, h, 1)
    xxnew, yynew = np.meshgrid(xnew, ynew)
    # print("dense here")
    znew = interpolate.bisplev(xxnew[:,0], yynew[0,:], tck)
    dense_map = znew

    return dense_map

def get_dense_map(H, xedges, yedges, shape=(1216, 1920)):
    H = H.T
    # print(H.shape)
    img = cv2.resize(H, (shape[1], shape[0]), interpolation = cv2.INTER_LINEAR)
    # print(img.max(), img.min())
    
    # cv2.imwrite("test_dense_map.png", img)
    return img

def get_neg_map(img, the):
    img[np.where(img<the)] = 0
    img[np.where(img>=the)] = 1
    return img

if __name__=="__main__":
    fn = "/shared/xudongliu/data/argoverse-tracking/argo_track/sample/t_pc/0000/000076.png"
    img = cv2.imread(fn)
    img = img[:,:,0]
    H, xedges, yedges = get_dense(img, 1920/40, 1200/40)
    print("here")
    dense_map = get_dense_map(H, xedges, yedges)
    s_img = dense_map * (255/dense_map.max())
    cv2.imwrite('dense_map_test.png', s_img)

    neg_map = get_neg_map(dense_map, 4)
    s_img = neg_map * (255/neg_map.max())
    cv2.imwrite("neg_map_test.png", s_img)
    # H = H.T
    # print(H.shape)
    # X, Y = np.meshgrid(xedges, yedges)
    # plt.pcolormesh(X, Y, H)
    # print("here")
    # plt.savefig("density_test.png")

