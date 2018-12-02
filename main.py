import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import glob
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.5f}'.format})

# My library
import calibration as cal
import camera_pose as cam
import plane_sweeping as pl

# Get current working directory and locate images
curr_work_dir = os.getcwd()
img_path = curr_work_dir + "/images/"
chessboard_path = img_path + "chessboard/"
scene_path = img_path + "scene/"

#-----------------------------------------------------------------#
#                              Part 1 
#-----------------------------------------------------------------#
# Chessboard pattern images
chessboard_imgs = glob.glob(chessboard_path +'*.jpg')

# Intrinsic matrix, K
ret, K, dist, rvecs, tvecs, mse = cal.camera_calibration()
print("K:\n", K)

# Radial distortion coefficients
print("\nRadial distortion coefficients:\n", dist)

# Reprojection mean square error
print("\nMean square error: ", "{:.5f}".format(mse))

#-----------------------------------------------------------------#
#                              Part 2 
#-----------------------------------------------------------------#
# Images of similar scene and w/ objects at different distances
scene_imgs = glob.glob(scene_path +'*.jpg')

#-----------------------------------------------------------------#
#                              Part 3 
#-----------------------------------------------------------------#
#relative_camera_pose()
img1 = scene_path + "myleft.jpg"
img2 = scene_path + "myright.jpg"

R1, R2, t = cam.relative_camera_pose(img1, img2, K, dist)

# Epipolar lines on the images

# Matrix R_R_L and r_R
print("\nRotation matrix left to right:\n", R1)
print("\nTranslation vector from right reference frame:\n", t)

# Re-projected feature points on the first image

#-----------------------------------------------------------------#
#                              Part 4 
#-----------------------------------------------------------------#
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html#py-depthmap

# d_min and d_max

# N=20 warped second images

# Resulting depth image, in grayscale
