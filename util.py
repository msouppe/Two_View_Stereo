import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Function: load_images
# Parameter(s): f_path
# Return: Array of images
# Description: Retrieves images from specified src
def load_images(f_path, roi=False):
	img_array = []
	imgs = os.listdir(f_path)
	print("load_images(): ", sorted(imgs))

	for image in sorted(imgs):
		if image.endswith(".JPG"):
			#img = PImage.open(f_path + image)
			img = cv.imread(f_path + image)				
			img_array.append(region_of_interest(img, roi))

	return img_array