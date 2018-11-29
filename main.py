import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
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
#chessboard_imgs = ut.load_images(chessboard_path)

# Intrinsic matrix, K
ret, mtx, dist, rvecs, tvecs, mse = cal.camera_calibration()
print("K:\n", mtx)

# Radial distortion coefficients
print("\nRadial distortion coefficients:\n", dist)

# Reprojection mean square error
print("\nMean square error: ", "{:.5f}".format(mse))

#-----------------------------------------------------------------#
#                              Part 2 
#-----------------------------------------------------------------#
# Images of similar scene and w/ objects at different distances


#-----------------------------------------------------------------#
#                              Part 3 
#-----------------------------------------------------------------#
# Epipolar lines on the images

# Matrix R_R_L and r_R

# Re-projected feature points on the first image

#-----------------------------------------------------------------#
#                              Part 4 
#-----------------------------------------------------------------#
# d_min and d_max

# N=20 warped second images

# Resulting depth image, in grayscale
