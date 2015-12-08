""" 
====================================================
    Faces recognition and detection using OpenCV 
====================================================

The dataset used is the Extended Yale Database B

  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html


Summary:
        Facial Recognition Using OpenCV and SVM

        Created by:  Chenxing Ouyang

"""

import cv2
import os
import numpy as np
from scipy import ndimage
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import utils as ut

print(__doc__)



IMAGE_DIM = (50,50) # h = 50, w = 50

target_names = ["Alex", "Ravi"]

# load YaleDatabaseB
for i in range (1, 35):
    missing_database = [14]
    name_prefix = "yaleB"
    if i < 10:
        name_index = "0" + str(i)
    else:
        name_index = str(i)
    name = name_prefix + name_index
    if i in missing_database:
        print name, " is missing"
    else:
        target_names.append(name)

# print target_names

X, y = ut.load_data(target_names, data_directory = "../face_data/")


for i in range(1,3): print ("\n")
print y.shape[0], " samples from ", len(target_names), " people are loaded"
for i in range(1,3): print ("\n")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150 # maximum number of components to keep

print("\nExtracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, IMAGE_DIM[0], IMAGE_DIM[1]))

# pca = RandomizedPCA(n_components=None, whiten=True).fit(X_train)
# eigenfaces = pca.components_.reshape((pca.components_.shape[0], IMAGE_DIM[0], IMAGE_DIM[1]))

print("\nProjecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test) 

# Train a SVM classification model

print("\nFitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
# Best Estimator found:
# bestEstimatorFound = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)

clf = clf.fit(X_train_pca, y_train)
print("\nBest estimator found by grid search:")
print(clf.best_estimator_)


# Quantitative evaluation of the model quality on the test set
print("\nPredicting people's names on the test set")
y_pred = clf.predict(X_test_pca)


print "predicated names: ", y_pred
print "actual names: ", y_test
print "Test Error Rate: ", ut.errorRate(y_pred, y_test)

###############################################################################
# Testing

X_test_pic1 = X_test[0]
X_test_pic1_for_display = np.reshape(X_test_pic1, IMAGE_DIM)

t0 = time()
X_test_pic1_pca = pca.transform(X_test_pic1)
pic1_pred = clf.predict(X_test_pic1_pca)
pic1_pred_name = target_names[pic1_pred]
print("Prediction took %0.3fs" % (time() - t0))


for i in range(1,3): print ("\n")
print "Testing picture_1 name: ", pic1_pred_name
for i in range(1,3): print ("\n")

# Display the picture
plt.figure(1)
plt.title(pic1_pred_name)
plt.subplot(111)
plt.imshow(X_test_pic1_for_display)
plt.show()

