import cv2 as cv
import numpy as np
import glob
#import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
from numba import jit

@jit
def fast_loop(img, xyz, uv ):
    tobii_depth = np.zeros(img.shape)
    for ii, uv in enumerate(uv):
        if ((0 < uv[1] < tobii_depth.shape[0] -1 ) and (0 < uv[0] < tobii_depth.shape[1] -1)):
            tobii_depth[int(uv[1]), int(uv[0])] = xyz[ii,2]

    return tobii_depth

def depth_to_tobii(tobii_img, rs_d_img, K_tobii, K_rs_d, dist_tobii, dist_rs_d, tobii_T_rs, debug = False):

    # Function returns an image similar to the tobii rgb image but with depths in mm

    mapx, mapy = cv.initUndistortRectifyMap( K_rs_d, dist_rs_d, None, None, rs_d_img.shape[::-1], 5)
    rs_d_img_undist = cv.remap(rs_d_img, mapx, mapy, cv.INTER_NEAREST)
    rs_d_xy = np.where(~np.isnan(rs_d_img_undist))
    rs_d_z = rs_d_img_undist[~np.isnan(rs_d_img_undist)].astype(float)
    rs_d_xy_h = np.stack((rs_d_xy[1].astype(float), rs_d_xy[0].astype(float), np.ones_like(rs_d_xy[1])), axis=1)
    rs_d_xy_n = (np.linalg.inv(K_rs_d) @ rs_d_xy_h.T).T
    rs_d_xyz = np.stack((rs_d_xy_n[:,0] * rs_d_z , rs_d_xy_n[:,1] * rs_d_z, rs_d_z), axis = 1)
    rs_d_xyz_h = np.hstack((rs_d_xyz, np.ones_like(rs_d_xyz)))[:,:4]
    tobii_xyz = (tobii_T_rs @ rs_d_xyz_h.T).T[:,:3]

    pc_n = tobii_xyz.copy()
    pc_n = tobii_xyz[:,:3] / tobii_xyz[:,2][:,None] # project into normalized image plane
    tobii_uv = (K_tobii@pc_n.T).T
    tobii_depth = fast_loop(tobii_img, tobii_xyz, tobii_uv)

    return tobii_depth


