import cv2 as cv
import numpy as np
import glob

# Soure https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
def camera_calibration():
	# termination criteria
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob.glob('images/chessboard/*.jpg')

	gray = None

	for fname in images:
		img = cv.imread(fname)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		
		# Find the chess board corners
		ret, corners = cv.findChessboardCorners(gray, (9,6),None)

		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
			imgpoints.append(corners)

			# Draw and display the corners
			cv.drawChessboardCorners(img, (9,6), corners, ret)
			cv.imshow('img',img)
			cv.waitKey(500)

	cv.destroyAllWindows()

	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	mse = mean_sqaure_error(imgpoints, objpoints, rvecs, tvecs, mtx, dist)

	return ret, mtx, dist, rvecs, tvecs, mse

def mean_sqaure_error(imgpoints, objpoints, rvecs, tvecs, mtx, dist):
	mean_error = 0
	for i in range(len(objpoints)):
	    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
	    mean_error += error

	return mean_error/len(objpoints)