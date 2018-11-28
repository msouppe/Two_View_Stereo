import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# My library
import util as ut

# Get current working directory and locate images
curr_work_dir = os.getcwd()
img_path = curr_work_dir + "/images/"
chessboard_path = img_path + "/chessboard/"
scene_path = img_path + "/scene/"

#-----------------------------------------------------------------#
#                              Part 1 
#-----------------------------------------------------------------#
# Chessboard pattern images

# Intrinsic matrix, K

# Radial distortion coefficients

# Reprojection mean square error

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
