"""
Auther: Chenxing Ouyang <c2ouyang@ucsd.edu>

This file is part of Cogs 109 Project.

Summary: Utilties used for facial tracking in OpenCV and facial recognition in SVM

"""


import cv2
import numpy as np
from scipy import ndimage
import os
import errno


###############################################################################
# Used For Facial Tracking and Traning in OpenCV

def rotate_image(image, angle, scale = 1.0):
    """ returns an rotated image with the same dimensions """
    if angle == 0: return image
    h, w = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

def trim(img, dim):
    """ dim = (y, x),  img.shape = (x, y) retruns a trimmed image with black paddings removed"""
    # if the img has the same dimension then do nothing
    if img.shape[0] == dim[1] and img.shape[1] == dim[0]: return img
    x = int((img.shape[0] - dim[1])/2) + 1
    y = int((img.shape[1] - dim[0])/2) + 1
    trimmed_img = img[x: x + dim[1], y: y + dim[0]]   # crop the image
    return trimmed_img

def clean_directory(directory = "../pics"):
    """ Deletes all files and folders contained in the directory """
    for the_file in os.listdir(directory):
        file_path = os.path.join(directory, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception, e:
            print e


def create_directory(path):
    """ create directories for saving images"""
    try:
        print "Making directory"
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def create_profile_in_database(profile_folder_name, database_path="../face_data/", clean_directory=False):
    """ Save to the default directory """
    profile_folder_path = database_path + profile_folder_name + "/"
    create_directory(profile_folder_path)
    # Delete all the pictures before recording new
    if clean_directory: 
        clean_directory(profile_folder_path) 
    return profile_folder_path




###############################################################################
# Used for Facial Recognition in SVM

def readImage(directory, y, dim = (50, 50)):
    """ Takes in a directory of images
      Returns X_data = (numOfFace X ImgPixelSize) face data array 
              Y_data = (numOfFace X 1) target_name_index array
    """
    X_data = np.array([])
    index = 0
    for the_file in os.listdir(directory):
        file_path = os.path.join(directory, the_file)
        if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".pgm"):
            img = cv2.imread(file_path, 0)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img_data = img.ravel()
            X_data = img_data if not X_data.shape[0] else np.vstack((X_data,img_data))
            index += 1

    Y_data = np.empty(index, dtype = int)
    Y_data.fill(y)
    return X_data, Y_data


def errorRate(pred, actual):
    """ Returns the error rate """
    if pred.shape != actual.shape: return None
    error_rate = np.count_nonzero(pred - actual)/float(pred.shape[0])
    return error_rate

def recognitionRate(pred, actual):
    """ Returns the recognition rate and error rate """
    if pred.shape != actual.shape: return None
    error_rate = np.count_nonzero(pred - actual)/float(pred.shape[0])
    recognitionRate = 1.0 - error_rate
    return recognitionRate, error_rate


def load_data(target_names, data_directory):
    """ Takes in a list of target_names (names of the directory that contains face pics)
      Retruns X_mat = (numbeOfFace X numberOfPixel) face data matrix 
              Y_mat = (numbeOfFace X 1) target_name_index matrix
    """
    if len(target_names) < 2: return None
    first_data = str(target_names[0])
    first_data_path = os.path.join(data_directory, first_data)
    X1, y1 = readImage(first_data_path, 0)
    X_mat = X1
    Y_mat = y1
    print "Loading Database: "
    print 0,"    ", first_data_path
    for i in range(1, len(target_names)):
        directory_name = str(target_names[i])
        directory_path = os.path.join(data_directory, directory_name)
        tempX, tempY = readImage(directory_path, i)
        X_mat = np.concatenate((X_mat, tempX), axis=0)
        Y_mat = np.append(Y_mat, tempY)
        print i, "    ", directory_path
    return X_mat, Y_mat

