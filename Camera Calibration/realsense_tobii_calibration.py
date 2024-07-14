import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import open3d
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

    if debug:
        visualise(rs_d_xyz)

    pc_n = tobii_xyz.copy()
    pc_n = tobii_xyz[:,:3] / tobii_xyz[:,2][:,None] # project into normalized image plane
    tobii_uv = (K_tobii@pc_n.T).T
    tobii_depth = fast_loop(tobii_img, tobii_xyz, tobii_uv)

    return tobii_depth



    print("finich")

def calibrate_camera(impts, ims, board):

    objp = np.zeros((board[0] * board[1] , 3), np.float32)
    objp[:,:2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1,2)
    objps = [objp for i in range(len(impts))]

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objps, impts, ims[0].shape[::-1], None, None, flags=cv.CALIB_FIX_ASPECT_RATIO)
    
    print("\nRMSE : {}".format(ret))
    print("K : ")
    print(K)
    print("Dist:")
    print(dist)
    if False:
        plt.imshow(cv.undistort(ims[0], K, dist))
        plt.show()

    return K, dist

def visualise(xyz_1):
    # Class for visualizing registration restults
    tar_pcd_o3d = open3d.geometry.PointCloud()
    tar_pcd_o3d.points = open3d.utility.Vector3dVector(xyz_1)
    tar_pcd_o3d.paint_uniform_color([0, 0, 1])  # BLUE
    open3d.visualization.draw_geometries([tar_pcd_o3d])



idxs = [0,1,2,5,6,7,8,10,11,12] #Some images did not show the whole checkerboard, on good images are included
board = (8,13) # checker board number of inner corners, 

folder = "E:\Prosjekter\Robofarmer\student_calib\synched_calibration 2\synched_calibration" # TODO change this

fns_ir = np.array(glob.glob(folder +  r"\intel\output_frames_small\ir_frame*"))[idxs]
fns_d = np.array(glob.glob(folder +  r"\intel\output_frames_small\depth_frame*"))[idxs]
fns_rgb = np.array(glob.glob(folder +  r"\intel\output_frames_small\color_frame*"))[idxs]
fns_tobii = np.array(glob.glob(folder +  r"\tobii\small chess\img*"))[idxs]

# CHECK THAT FILES ARE IN SAME ORDER i.e. tobii_im 1 <--> rs_im 1 etc.

print("CHECK THAT FILES ARE IN THE SAME ORDER\n")
[print(os.path.split(fn)[1]) for fn in fns_ir]
print("\n")
[print(os.path.split(fn)[1]) for fn in fns_tobii]

# read images
imgs_ir = [np.load(fn) for fn in fns_ir]
imgs_d = [np.load(fn) for fn in fns_d]
imgs_rgb = [cv.imread(fn, 0) for fn in fns_rgb]
imgs_tobii = [cv.imread(fn, 0) for fn in fns_tobii]

corners_ir = []
corner_depths_ir = []
corners_rgb = []
corners_tobii = []


# ********** CALIBRATION START **********
# corner detection
for i in range(len(fns_ir)):
    if (cv.findChessboardCorners(imgs_ir[i], board, None )[0]):
        corners_ir += [cv.findChessboardCorners(imgs_ir[i], board, None )[1]]
        corners_rgb += [cv.findChessboardCorners(imgs_rgb[i], board, None )[1]]
        corners_tobii += [cv.findChessboardCorners(imgs_tobii[i], board, None )[1]]
        corner_ir_idx = [np.round(p).astype(int)[0]  for p in corners_ir[0]]
        corner_depths_ir += [imgs_d[i][pp[1], pp[0]] for pp in corner_ir_idx ] 

        # DEBUG / check corner detections
        if False:
            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(cv.drawChessboardCorners(imgs_ir[i], board, corners_ir[-1], True))
            ax[0,1].imshow(cv.drawChessboardCorners(imgs_d[i], board, corners_ir[-1], True))
            ax[1,0].imshow(cv.drawChessboardCorners(imgs_rgb[i], board, corners_rgb[-1], True))
            ax[1,1].imshow(cv.drawChessboardCorners(imgs_tobii[i], board, corners_tobii[-1], True))
            plt.show()

    else:
        print("Could not find corners at idx : {}".format(i))

# intrinsics
K_ir, dist_ir = calibrate_camera(corners_ir, imgs_ir, board)
K_tobii, dist_tobii = calibrate_camera(corners_tobii, imgs_tobii, board)

xyz_ir = []
xy_tobii = []

# project RS corners to XYZ 
for i in range(len(fns_ir)):
    # xyz from IR / Depth image
    xy_n_ir = np.array(cv.undistortImagePoints(corners_ir[i], K_ir, dist_ir)).squeeze()
    xy_nh_ir = np.stack((xy_n_ir[:,0], xy_n_ir[:,1], np.ones_like(xy_n_ir[:,0])), axis = 1) 
    xy_nh_ir = np.linalg.inv(K_ir) @ xy_nh_ir.T # this shouldnt be neccesary but is..
    xyz = (xy_nh_ir * np.array(corner_depths_ir[i])).T
    xyz_ir += [xyz]

    # xy from tobii camera
    xy_tobii += [np.array(cv.undistortImagePoints(corners_tobii[i], K_tobii, dist_tobii)).squeeze()]

# Calculate tranform 
A = np.array(xyz_ir).reshape(-1, 3)
B = np.array(corners_tobii).reshape(-1, 2)
ret, rvec, tvec, inliers = cv.solvePnPRansac(np.array(A, dtype=float), np.array(B, dtype = float), K_tobii, dist_tobii, None, None, False, 10000, 1)

T = np.eye(4)
T[:3,:3] = cv.Rodrigues(rvec)[0] 
T[:3, 3] = np.squeeze(tvec)
print(" T :")
print(T)

# Check reprojected corner points - if points are aligned --> Calibration is good
if False:
    for i in range(len(fns_ir)):

    # Check if calibration looks ok, RS corners in tobii image:
        xyz_h = np.hstack((xyz_ir[i], np.ones_like(xyz_ir[i])))[:,:4]
        xyz_tobii = (T@xyz_h.T).T
        xy_n_tobii = xyz_tobii / (xyz_tobii[:,2])[:, None]
        uv_tobii = ((K_tobii@xy_n_tobii[:,:3].T).T)[:,:2]
        corners_tobii_reproj = uv_tobii.astype(np.float32)[:,None, :]
        im_tobii_undist = cv.undistort(imgs_tobii[i], K_tobii, dist_tobii)
        im_tobii_undist = np.stack((im_tobii_undist,im_tobii_undist,im_tobii_undist),axis = 2)
        im = cv.drawChessboardCorners(im_tobii_undist, board, corners_tobii_reproj, True)
        corners_tobii_undist = cv.undistortImagePoints(corners_tobii[i], K_tobii, dist_tobii)
        im = cv.drawChessboardCorners(im, board, corners_tobii_undist, True)
        plt.figure()
        plt.imshow(im)
        plt.show()


# ********** CALIBRATION END **********



# ********** DEPTH TO TOBII START **********

# make depth image in Tobii frame
for i in range(len(fns_ir)):

    tobii_depth = depth_to_tobii(imgs_tobii[i], imgs_d[i], K_tobii, K_ir, dist_tobii, dist_ir, T) # depth in mm,

    if True:
        depth_disp = tobii_depth.copy()
        depth_disp[depth_disp > 1000] = 1000
        imim = np.stack((imgs_tobii[i], (depth_disp/4), np.zeros_like(tobii_depth)), axis = 2)
        plt.imshow(imim.astype(np.uint8))
        plt.show()

# ********** DEPTH TO TOBII END **********

print("finish")
