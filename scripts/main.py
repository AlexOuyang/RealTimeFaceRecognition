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


import utils as ut
import svm

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

# Build the classifier
face_data, face_target = ut.load_data(target_names, data_directory = "../face_data/")
clf = svm.build_SVC(face_data, face_target)


###############################################################################
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
pic1_pred_name = svm.predict(clf, X_test_pic1, target_names)
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
