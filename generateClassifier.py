#!/usr/bin/python

# Import the modules
import joblib
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from collections import Counter

# Load the dataset
dataset = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

# Extract the HOG features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)),
             orientations=9,
             pixels_per_cell=(14, 14),
             cells_per_block=(1, 1),
             visualize=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)

print("Count of digits in dataset:", Counter(labels))

# Create a linear SVM object
clf = LinearSVC(max_iter=10000)  # added max_iter to ensure convergence

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump((clf, pp), "digits_cls.pkl", compress=3)
