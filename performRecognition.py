#!/usr/bin/python

# Import the modules
import cv2
import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required=True)
parser.add_argument("-i", "--image", help="Path to Image", required=True)
args = vars(parser.parse_args())

# Load the classifier
clf, pp = joblib.load(args["classiferPath"])

# Read the input image 
im = cv2.imread(args["image"])

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
_, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image (handle OpenCV 3.x/4.x compatibility)
contours = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Get rectangles containing each contour
rects = [cv2.boundingRect(ctr) for ctr in contours]

# For each rectangular region, calculate HOG features and predict
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), 
                       (rect[0] + rect[2], rect[1] + rect[3]), 
                       (0, 255, 0), 3) 

    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, np.ones((3, 3), np.uint8))

    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, 
                          pixels_per_cell=(14, 14), 
                          cells_per_block=(1, 1), 
                          visualize=False)
    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))

    # Predict digit
    nbr = clf.predict(roi_hog_fd)
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), 
                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

# Show result
cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
