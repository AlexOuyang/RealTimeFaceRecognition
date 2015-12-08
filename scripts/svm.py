"""
Auther: Chenxing Ouyang <c2ouyang@ucsd.edu>

This file is part of Cogs 109 Project.

"""

import cv2
import os
import numpy as np
from scipy import ndimage
from time import time

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


def build_SVC(face_data, face_target):
    """ Build SVM classification modle using the face_data matrix (numOfFace X numOfPixel)
        and face_target array
        Returns the SVM classification modle
    """
    X = face_data
    y = face_target

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

    return clf


def predict(clf, img, target_names):
    """ Takes in a classifier, img (1 X w*h) and target_names
        Returns the predicated name
    """
    principle_component = pca.transform(X_test_pic1)
    pred = clf.predict(principle_component)
    name = target_names[pred]
    return name


