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

	# Undistort
	dst = cv.undistort(img, mtx, dist, None, newcameramtx)
	# dst = cv.undistortPoints(img, mtx, dist, None, newcameramtx)

	cv.imwrite(img_name, dst)

	print('undistort() complete!')
	return dst

# Source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
# Create “feature points” in each image:
def create_feature_points(img1, img2, mtx, dist):
	sift = cv.xfeatures2d.SIFT_create()

	# Find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	return kp1, des1, kp2, des2

# Match the feature points across the two images:
def match_feature_points(kp1, des1, kp2, des2):
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

	# Compute the fundamental matrix F and find some epipolar lines
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

	# We select only inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]

	return pts1, pts2, F

# Draw epilines
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

def draw_image_epipolar_lines(img1, img2, pts1, pts2, F):
	lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
	lines1 = lines1.reshape(-1,3)
	img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

	lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

	plt.subplot(121),plt.imshow(img5)
	plt.subplot(122),plt.imshow(img3)
	plt.show()

def relative_camera_pose(img1_, img2_, K, dist, undistort_=False):

	if undistort_:
		img1 = undistort(img1_, 'myleft_undistort.jpg', K, dist)
		img2 = undistort(img2_, 'myright_undistort.jpg', K, dist)
	else:
		img1 = cv.imread(img1_, 0)
		img2 = cv.imread(img2_, 0)

	kp1, des1, kp2, des2 = create_feature_points(img1, img2, K, dist)
	pts1, pts2, F = match_feature_points(kp1, des1, kp2, des2)

	draw_image_epipolar_lines(img1, img2, pts1, pts2, F)

	# Compute the essential matrix E
	# Help: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
	E, _ = cv.findEssentialMat(pts1, pts2, K)
	print("\nEssential Matrix, E:\n", E)

	# Decompose the essential matrix into R, r
	R1, R2, t = cv.decomposeEssentialMat(E)
	print("\nRotation matrix left to right:\n", R1)
	print("\nTranslation vector from right reference frame:\n", t)

	# Re-projected feature points on the first image
	# http://answers.opencv.org/question/173969/how-to-give-input-parameters-to-triangulatepoints-in-python/
	x = np.array([0,0,0])
	zero_vect = x.reshape(3,1)
	P1 = np.append(R1, zero_vect, 1)
	P2 = np.append(R2, t, 1)
	print("End of relative_camera_pose()")

	# Calculate the depth of the matching points:
	# http://answers.opencv.org/question/117141/triangulate-3d-points-from-a-stereo-camera-and-chessboard/
	# https://stackoverflow.com/questions/22334023/how-to-calculate-3d-object-points-from-2d-image-points-using-stereo-triangulatio/22335825
	# points4D = cv.triangulatePoints(P1, np.transpose(pts1))
	# triangulatePoints()

	return R1, R2, t