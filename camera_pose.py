import os
import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt

# Soure https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
# 1. Undistort the images using the parameters you found during calibration
# undistortPoints()
def undistort(image, img_name, mtx, dist):
	img = cv.imread(image, 0)
	h, w = img.shape[:2]
	print("height, width: ", h, w)
	newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

	print("newcameramtx:", newcameramtx)

	# Undistort
	dst = cv.undistortPoints(img, mtx, dist, None, newcameramtx)
    
    # Crop the image
	# x,y,w,h = roi
	# print(roi)
	# dst = dst[y:y+h, x:x+w]
	cv.imwrite(img_name, dst)

	print('undistort complete!')
	return dst

# Source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
# 2. Create “feature points” in each image:
# 	sift = cv.xfeatures2d.SIFT_create()
# 	sift.detectAndCompute()
def create_feature_points(img1_, img2_, mtx, dist):

	# img1 = undistort(img1_, 'myleft_undistort.jpg', mtx, dist)
	# img2 = undistort(img2_, 'myright_undistort.jpg', mtx, dist)
	
	img1 = cv.imread(img1_, 0)
	img2 = cv.imread(img2_, 0)

	sift = cv.xfeatures2d.SIFT_create()

	print (img1.dtype)
	print (img2.dtype)

	# Find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	print('create_feature_points complete!')
	return kp1, des1, kp2, des2

# 3. Match the feature points across the two images:
# 	cv.FlannBasedMatcher()
# 	flann.knnMatch()
def match_feature_points(img1_, img2_, mtx, dist):
	kp1, des1, kp2, des2 = create_feature_points(img1_, img2_, mtx, dist)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	good = []
	pts1 = []
	pts2 = []

	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.8*n.distance:
			good.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)

	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

	# We select only inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]
	print('inside match_feature_points()')
	return pts1, pts2, F

# # Draw epilines
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1, img2




# # 4. Compute the fundamental matrix F and find some epipolar lines (note: although this
# # 	is not strictly necessary for our purpose, you still are expected to do it):
# # 	findFundamentalMat()
# # 	computeCorrespondEpilines()

# # 5. Compute the essential matrix E:
# # findEssentialMat()

# # 6. Decompose the essential matrix into R, r
# # decomposeEssentialMat()

# # 7. Calculate the depth of the matching points:
# # triangulatePoints()









# def relative_camera_pose(img1, img2):


