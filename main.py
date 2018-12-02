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
scene_imgs = glob.glob(scene_path +'*.HEIC')

#-----------------------------------------------------------------#
#                              Part 3 
#-----------------------------------------------------------------#
#relative_camera_pose()
img1 = scene_path + "myleft.jpg"
img2 = scene_path + "myright.jpg"
pts1, pts2, F = cam.match_feature_points(img1, img2, K, dist)
print("pts1.shape", pts1.shape)

# Epipolar lines on the images
gray1 = cv.imread(img1, 0)
gray2 = cv.imread(img2, 0)
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
lines1 = lines1.reshape(-1,3)
img5,img6 = cam.drawlines(gray1,gray2,lines1,pts1,pts2)

lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = cam.drawlines(gray2,gray1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

# Matrix R_R_L and r_R
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
E, _ = cv.findEssentialMat(pts1, pts2, K)
print("\nEssential Matrix, E:\n", E)

R1, R2, t = cv.decomposeEssentialMat(E)
print("\nRotation matrix left to right:\n", R1)
print("\nTranslation vector from right reference frame:\n", t)

# Re-projected feature points on the first image
# http://answers.opencv.org/question/173969/how-to-give-input-parameters-to-triangulatepoints-in-python/
x = np.array([0,0,0])
zero_vect = x.reshape(3,1)
P1 = np.append(R1, zero_vect, 1)
P2 = np.append(R2, t, 1)
print("\nP1:\n", P1)
print("\nP2:\n", P2)
# http://answers.opencv.org/question/117141/triangulate-3d-points-from-a-stereo-camera-and-chessboard/
# https://stackoverflow.com/questions/22334023/how-to-calculate-3d-object-points-from-2d-image-points-using-stereo-triangulatio/22335825
#points4D = cv.triangulatePoints(P1, np.transpose(pts1))
#print(points4D)
#-----------------------------------------------------------------#
#                              Part 4 
#-----------------------------------------------------------------#
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html#py-depthmap

# d_min and d_max

# N=20 warped second images

# Resulting depth image, in grayscale
