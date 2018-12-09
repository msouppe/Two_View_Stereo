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
	dst = cv.undistort(img, mtx, dist, None, None)

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
		if m.distance < 0.4*n.distance:
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
	plt.savefig('plot_epilines.png')
	plt.show()

def relative_camera_pose(img1_, img2_, K, dist):

	print("\n\nRelativeCamPose:\n")

	img1 = undistort(img1_, 'left_undistort.jpg', K, dist)
	img2 = undistort(img2_, 'right_undistort.jpg', K, dist)

	kp1, des1, kp2, des2 = create_feature_points(img1, img2, K, dist)
	pts1, pts2, F = match_feature_points(kp1, des1, kp2, des2)

	draw_image_epipolar_lines(img1, img2, pts1, pts2, F)

	# Compute the essential matrix E
	# Help: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
	E, _ = cv.findEssentialMat(pts1, pts2, K)
	print("\nEssential Matrix, E:\n", E)

	# Decompose the essential matrix into R, r
	R1, R2, t = cv.decomposeEssentialMat(E)

	# Re-projected feature points on the first image
	# http://answers.opencv.org/question/173969/how-to-give-input-parameters-to-triangulatepoints-in-python/
	print("K:\n", K)
	print("dist: \n", dist)

	print("R1:\n", R1)
	print("R2:\n", R2)
	print("t\n", t)

	R2_t = np.hstack((R2,(t)))
	print("R2_t:\n", R2_t)
	
	zero_vector = np.array([[0,0,0]])
	zero_vector = np.transpose(zero_vector)

	# Create projection matrices P1 and P2
	P1 = np.hstack((K,zero_vector))
	P2 = np.dot(K,R2_t)

	print("P1:\n", P1)
	print("P2:\n", P2)

	pts1 = pts1.astype(np.float)
	pts2 = pts2.astype(np.float)

	pts1 = np.transpose(pts1)
	pts2 = np.transpose(pts2)

	points4D = cv.triangulatePoints(P1, P2, pts1, pts2)
	aug_points3D = points4D/points4D[3]
	print("\nAugmented points3D:\n",aug_points3D)

	min_depth = min(aug_points3D[2])
	max_depth = max(aug_points3D[2])
	print("\n\nMin depth:\n", min_depth)
	print("Max depth:\n\n", max_depth)

	N = (max_depth-min_depth)/20
	print("N=20:\n", N)


	equispaced_dist = np.linspace(min_depth, max_depth, num=20)
	print("Equispaced distance:\n", equispaced_dist)

	projectPoints = np.dot(P1, aug_points3D)
	print("projectPoints:\n", projectPoints)

	points2D = projectPoints[:2]
	print("\n2DPoints:\n", points2D)
	
	# Calculate the depth of the matching points:
	# http://answers.opencv.org/question/117141/triangulate-3d-points-from-a-stereo-camera-and-chessboard/
	# https://stackoverflow.com/questions/22334023/how-to-calculate-3d-object-points-from-2d-image-points-using-stereo-triangulatio/22335825
	
	homography = []
	output_warp = []

	for i in range(0,20):
		nd_vector = np.array([0,0,-1,equispaced_dist[i]])
		
		P1_aug = np.vstack((P1,nd_vector))
		P2_aug = np.vstack((P2,nd_vector))
		
		#print("P1_aug:\n",P1_aug)
		#print("P2_aug:\n",P2_aug)
		
		P2_inv = np.linalg.inv(P2_aug)  
		#print("P2_inv:\n", P2_inv)

		P1P2_inv = np.dot(P1_aug, P2_inv)
		#print("P1P2_inv:\n",P1P2_inv)

		R =  P1P2_inv[:3,:3]
		#print("R:\n", R)
		#KR = np.dot(K,R)
		#homography = np.dot(KR,K_inv)

		homography.append(R)
		#print("\nHomography" + str(i) + ":")
		print("\n\nhomography" + str(i))
		print(homography[i])
		
		output_warp.append(cv.warpPerspective(img2, homography[i], None))
		cv.imwrite('Warped_output_' + str(i) + '.jpg', output_warp[i])

	#h,w = img2.shape

	#output_warp = cv.warpPerspective(img2, homography, None)
	#cv.imwrite('Warped_output.jpg',output_warp)




	return R1, R2, t