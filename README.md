# RealTimeFacialRecognition

Real time facial tracking and recognition using harrcascade and SVM(Radial Basis Function Kernal). Designed SVM classification model (radial basis functional kernel) using cross-validation and grid search, implemented using scikit in python, and trained on Extended Yale Database B. Achieved facial Tracking in OpenCV and optimized Haarcascade to detect up to 45 degrees head tilting.

The first part of the system is facial detection, which is achieved using Haar feature-based cascade classifiers, a novel way proposed by Paul Viola and Michael Jones in their 2001 paper, “Rapid Object Detection using a Boosted Cascade of Simple Features”. To further improve the method, geometric transformations are applied to each frame for face detection, allowing detection up to 45 degrees of head tilting. The second part of the system, face recognition, is achieved through a hybrid model consisting of feature extraction and classification trained on the cropped Extended Yale Face Database B. To build the model, 2452  samples from  38  people in the database are splitted into training and testing sets by a ratio of 3:1. The top 150 eigenfaces are extracted from 1839 training faces in the database using Principal Component Analysis (PCA). The principal components are then feeded into the C-SVM Classification model and trained with various kernel tricks. At the end of the recognition task, an accuracy of 93.3% is obtained with the Radial Basis Function (RBF) kernel on the testing set of 613 samples. Used in the real time application via webcam, the proposed system runs at 10 frames per second with high recognition accuracy relative to the number of training images of real time testers and how representative those training images are. 



The dataset used is the Extended Yale Face Database B Cropped

  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html


To Train: 
    
    python train.py FACE_NAME

FACE_NAME is the name of the user profile directory that you want to create in the default face_data folder for storing user face images and training the SVM classification model.

To Run:

    python main.py


Auther: Chenxing Ouyang <c2ouyang@ucsd.edu>
