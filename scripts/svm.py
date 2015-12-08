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

import utils as ut


def test_SVM(face_data, face_target, img_dim, target_names):
    """ Testing SVM

        Build SVM classification modle using the face_data matrix (numOfFace X numOfPixel)
        and face_target array, img_dim is a tuple of the dimension of each image(h,w)
        Returns the SVM classification modle
    """
    X = face_data
    y = face_target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150 # maximum number of components to keep

    print("\nExtracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))

    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, img_dim[0], img_dim[1]))

    # This portion of the code is used if the data is scarce, it uses the number 
    # of imputs as the number of features
    # pca = RandomizedPCA(n_components=None, whiten=True).fit(X_train)
    # eigenfaces = pca.components_.reshape((pca.components_.shape[0], img_dim[0], img_dim[1]))

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

    ###############################################################################
    # Quantitative evaluation of the model quality on the test set
    print("\nPredicting people's names on the test set")
    y_pred = clf.predict(X_test_pca)
    # print "predicated names: ", y_pred
    # print "actual names: ", y_test
    print "Test Error Rate: ", ut.errorRate(y_pred, y_test)

    ###############################################################################
    # Testing

    X_test_pic1 = X_test[0]
    X_test_pic1_for_display = np.reshape(X_test_pic1, img_dim)

    t0 = time()
    pic1_pred_name = predict(clf, pca, X_test_pic1, target_names)
    print("\nPrediction took %0.3fs" % (time() - t0))


    for i in range(1,3): print ("\n")
    print "Testing picture_1 name: ", pic1_pred_name
    for i in range(1,3): print ("\n")

    # Display the picture
    # plt.figure(1)
    # plt.title(pic1_pred_name)
    # plt.subplot(111)
    # plt.imshow(X_test_pic1_for_display)
    # plt.show()

    return clf, pca


def build_SVC(face_data, face_target, img_dim):
    """ Build SVM classification modle using the face_data matrix (numOfFace X numOfPixel)
        and face_target array, img_dim is a tuple of the dimension of each image(h,w)
        Returns the SVM classification modle
    """
    X = face_data
    y = face_target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150 # maximum number of components to keep

    print("\nExtracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))

    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, img_dim[0], img_dim[1]))

    # This portion of the code is used if the data is scarce, it uses the number 
    # of imputs as the number of features
    # pca = RandomizedPCA(n_components=None, whiten=True).fit(X_train)
    # eigenfaces = pca.components_.reshape((pca.components_.shape[0], img_dim[0], img_dim[1]))

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
    # print "predicated names: ", y_pred
    # print "actual names: ", y_test
    print "Test Error Rate: ", ut.errorRate(y_pred, y_test)

    return clf, pca


def predict(clf, pca, img, target_names):
    """ Takes in a classifier, img (1 X w*h) and target_names
        Returns the predicated name
    """
    principle_component = pca.transform(img)
    pred = clf.predict(principle_component)
    name = target_names[pred]
    return name


