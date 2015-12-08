""" 
====================================================
    Faces recognition and detection using OpenCV 
====================================================

The dataset used is the Extended Yale Database B

  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html


Summary:
        Real time facial tracking and recognition using harrcascade 
        and SVM

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


###############################################################################
# Building SVC from database

# Used to load data bases
def load_Yale_Exteded_Database(number_of_Faces):
    for i in range (1, number_of_Faces):
        missing_database = [14]
        name_prefix = "yaleB"
        if i < 10:
            name_index = "0" + str(i)
        else:
            name_index = str(i)
        name = name_prefix + name_index
        if i in missing_database:
            print "Missing Database: ", name
        else:
            target_names.append(name)


IMAGE_DIM = (50,50) # h = 50, w = 50

target_names = ["Alex"]

# load YaleDatabaseB
load_Yale_Exteded_Database(10)

# print target_names

# Build the classifier
face_data, face_target = ut.load_data(target_names, data_directory = "../../../face_data/")

print face_target.shape[0], " samples from ", len(target_names), " people are loaded"
for i in range(1,2): print ("\n")

# clf = svm.build_SVC(face_data, face_target, IMAGE_DIM)
clf, pca = svm.test_SVM(face_data, face_target, IMAGE_DIM, target_names)


###############################################################################
# Facial Recognition and Tracking Live


FACE_DIM = (200, 200)
SKIP_FRAME = 2      # the fixed skip frame
frame_skip_rate = 0 # skip SKIP_FRAME frames every other frame
SCALE_FACTOR = 4 # used to resize the captured frame for face detection for faster processing speed
face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml") #create a cascade classifier
sideFace_cascade = cv2.CascadeClassifier('../data/haarcascade_profileface.xml')

# dictionary mapping used to keep track of head rotation maps
rotation_maps = {
    "left": np.array([-30, 0, 30]),
    "right": np.array([30, 0, -30]),
    "middle": np.array([0, -30, 30]),
}

def get_rotation_map(rotation):
    """ Takes in an angle rotation, and returns an optimized rotation map """
    if rotation > 0: return rotation_maps.get("right", None)
    if rotation < 0: return rotation_maps.get("left", None)
    if rotation == 0: return rotation_maps.get("middle", None)

current_rotation_map = get_rotation_map(0) 


webcam = cv2.VideoCapture(0)

ret, frame = webcam.read() # get first frame
frame_scale = (frame.shape[1]/SCALE_FACTOR,frame.shape[0]/SCALE_FACTOR)  # (y, x)

crop_face = []
num_of_face_saved = 0


while ret:
    key = cv2.waitKey(1)
    # exit on 'q' 'esc' 'Q'
    if key in [27, ord('Q'), ord('q')]: 
        break
    # resize the captured frame for face detection to increase processing speed
    resized_frame = cv2.resize(frame, frame_scale)

    processed_frame = resized_frame
    # Skip a frame if the no face was found last frame
    if frame_skip_rate == 0:
        faceFound = False
        for rotation in current_rotation_map:

            rotated_frame = ndimage.rotate(resized_frame, rotation)

            gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

            # return tuple is empty, ndarray if detected face
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            ) 

            # If frontal face detector failed, use profileface detector
            faces = faces if len(faces) else sideFace_cascade.detectMultiScale(                
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            # for f in faces:
            #     x, y, w, h = [ v*SCALE_FACTOR for v in f ] # scale the bounding box back to original frame size
            #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
            #     cv2.putText(frame, "DumbAss", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

            if len(faces):
                for f in faces:
                    # Crop out the face
                    x, y, w, h = [ v for v in f ] # scale the bounding box back to original frame size
                    crop_face = rotated_frame[y: y + h, x: x + w]   # img[y: y + h, x: x + w]
                    crop_face = cv2.resize(crop_face, FACE_DIM, interpolation = cv2.INTER_AREA)

                    # Name Prediction
                    face_to_predict = cv2.resize(crop_face, IMAGE_DIM, interpolation = cv2.INTER_AREA)
                    face_to_predict = cv2.cvtColor(face_to_predict, cv2.COLOR_BGR2GRAY)
                    face_to_predict = face_to_predict.ravel()
                    name_to_display = svm.predict(clf, pca, face_to_predict, target_names)

                    # Display frame
                    cv2.rectangle(rotated_frame, (x,y), (x+w,y+h), (0,255,0))
                    cv2.putText(rotated_frame, name_to_display, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

                # rotate the frame back and trim the black paddings
                processed_frame = ut.trim(ut.rotate_image(rotated_frame, rotation * (-1)), frame_scale)

                # reset the optmized rotation map
                current_rotation_map = get_rotation_map(rotation)

                faceFound = True


                break

        if faceFound: 
            frame_skip_rate = 0
            print "Face Found"
        else:
            frame_skip_rate = SKIP_FRAME
            print "Face Not Found"

    else:
        frame_skip_rate -= 1
        print "Face Not Found"


  
    cv2.putText(processed_frame, "Press ESC or 'q' to quit.", (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    cv2.imshow("Real Time Facial Recognition", processed_frame)



    if len(crop_face):
        cv2.imshow("Face", crop_face)
        # face_to_predict = cv2.resize(crop_face, IMAGE_DIM, interpolation = cv2.INTER_AREA)
        # face_to_predict = cv2.cvtColor(face_to_predict, cv2.COLOR_BGR2GRAY)
        # name_to_display = svm.predict(clf, pca, face_to_predict, target_names)
    # get next frame
    ret, frame = webcam.read()


webcam.release()
cv2.destroyAllWindows()


