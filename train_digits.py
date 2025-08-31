#!/usr/bin/python

import joblib
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from collections import Counter

print("🔄 Fetching MNIST dataset (70,000 samples)...")
dataset = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract features and labels
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

print("✅ Dataset loaded:", features.shape, "samples")

# Extract HOG features
print("🔄 Extracting HOG features...")
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)),
             orientations=9,
             pixels_per_cell=(14, 14),
             cells_per_block=(1, 1),
             visualize=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Normalize features
print("🔄 Normalizing features...")
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)

print("📊 Digit counts:", Counter(labels))

# Train classifier
print("🔄 Training LinearSVC (this may take a while)...")
clf = LinearSVC(max_iter=10000)
clf.fit(hog_features, labels)

# Save classifier
joblib.dump((clf, pp), "digits_cls.pkl", compress=3)
print("🎉 Training complete! Saved model as digits_cls.pkl")
