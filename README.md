# RealTimeFacialRecognition
Built with Python 2.7, OpenCV2, Numpy, Scipy, scikit-learn, matplotlib.

* note: this application currently only works on Mac because of forward slash in path is incompatible with windows

Summary
--------------
Real time facial tracking and recognition using harrcascade and SVM(Radial Basis Function Kernal). Designed SVM classification model using cross-validation and exhaustively grid search, implemented using scikit in python, and trained on Extended Yale Database B. Achieved facial Tracking in OpenCV and optimized Haarcascade to detect up to 45 degrees head tilting. To build the model, 2452  samples from  38  people in the database are splitted into training and testing sets by a ratio of 3:1. The top 150 eigenfaces are extracted from 1839 training faces in the database using Principal Component Analysis (PCA). The principal components are then feeded into the C-SVM Classification model and trained with various kernel tricks. At the end of the recognition task, an accuracy of 93.3% is obtained with the Radial Basis Function (RBF) kernel on the testing set of 613 samples. 

The dataset used is the Extended Yale Face Database B Cropped

  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html


Requirements
--------------
-  Install `OpenCV` on mac: http://www.mobileway.net/2015/02/14/install-opencv-for-python-on-mac-os-x/
-  To install other dependencies, cd into ./scripts/ then run: ``pip install -r requirements.txt``


Use as local command-line utility
---------------------------------


    $ git clone https://github.com/AlexOuyang/RealTimeFaceRecognition.git


Training For Face Recognition
-------------------------------

Training for face recognition using the command below. face_profile_name is the name of the user face profile directory that you want to create in the default ../face_profiles/ folder for storing user face images and training the SVM classification model:


    python train.py [face_profile_name=<the name of the profile folder in database>]


Example to create a face profile named David:


    python train.py David



Usage during run time:


    press and hold 'p' to take pictures of you continuously once a cropped face is detected from a 
    pop up window. All images are saved under ../face_profiles/face_profile_name

    press 'q' or 'ESC' to quit the application


Running Face Recognition In Real Time
--------------------------------------

Running the program in real time to recognize faces:


    python main.py


Or running with options (By default, scale_multiplier = 4):


    python main.py [scale_multiplier=<full screensize divided by scale_multiplier>]


Say you want to run with 1/2 of the full sreen size, specify that scale_multiplier = 4:

    python main.py 4



Auther: Chenxing Ouyang <c2ouyang@ucsd.edu>
