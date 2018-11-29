import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# My library
import util as ut

def camera_calibration(imgs):
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = ut.load_images(imgs)

	for fname in images:
		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners)

			# Draw and display the corners
			cv2.drawChessboardCorners(img, (7,6), corners2,ret)
			cv2.imshow('img',img)
			cv2.waitKey(500)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
	return [ret, mtx, dist, rvecs, tvecs]


def mean_sqaure_error(objpoints):
	mean_error = 0
	for i in xrange(len(objpoints)):
	    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	    tot_error += error

	print "total error: ", mean_error/len(objpoints)

	return mean_error/len(objpoints