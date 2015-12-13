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
import sys
import logging
import shutil


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

def read_images_from_single_face_profile(face_profile, face_profile_name_index, dim = (50, 50)):
    """
    Reads all the images from one specified face profile into ndarrays

    Parameters
    ----------
    face_profile: string
        The directory path of a specified face profile

    face_profile_name_index: int
        The name corresponding to the face profile is encoded in its index

    dim: tuple = (int, int)
        The new dimensions of the images to resize to

    Returns
    -------
    X_data : numpy array, shape = (number_of_faces_in_one_face_profile, face_pixel_width * face_pixel_height)
        A face data array contains the face image pixel rgb values of all the images in the specified face profile 

    Y_data : numpy array, shape = (number_of_images_in_face_profiles, 1)
        A face_profile_index data array contains the index of the face profile name of the specified face profile directory

    """
    X_data = np.array([])
    index = 0
    for the_file in os.listdir(face_profile):
        file_path = os.path.join(face_profile, the_file)
        if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".pgm"):
            img = cv2.imread(file_path, 0)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img_data = img.ravel()
            X_data = img_data if not X_data.shape[0] else np.vstack((X_data,img_data))
            index += 1

    if index == 0 : 
        shutil.rmtree(face_profile)
        logging.error("\nThere exists face profiles without images")

    Y_data = np.empty(index, dtype = int)
    Y_data.fill(face_profile_name_index)
    return X_data, Y_data

def delete_empty_profile(face_profile_directory):
    """
    Deletes empty face profiles in face profile directory and logs error if face profiles contain too little images

    Parameters
    ----------
    face_profile_directory: string
        The directory path of the specified face profile directory

    """
    for face_profile in os.listdir(face_profile_directory):
        if "." not in str(face_profile):
            face_profile = os.path.join(face_profile_directory, face_profile)
            index = 0
            for the_file in os.listdir(face_profile):
                file_path = os.path.join(face_profile, the_file)
                if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".pgm"):
                    index += 1
            if index == 0 : 
                shutil.rmtree(face_profile)
                print "\nDeleted ", face_profile, " because it contains no images"
            if index <= 2 : 
                logging.error("\nFace profile " + str(face_profile) + " contains too little images (At least 2 images are needed)")


def load_training_data(face_profile_directory):
    """
    Loads all the images from the face profile directory into ndarrays

    Parameters
    ----------
    face_profile_directory: string
        The directory path of the specified face profile directory

    face_profile_names: list
        The index corresponding to the names corresponding to the face profile directory

    Returns
    -------
    X_data : numpy array, shape = (number_of_faces_in_face_profiles, face_pixel_width * face_pixel_height)
        A face data array contains the face image pixel rgb values of all face_profiles

    Y_data : numpy array, shape = (number_of_face_profiles, 1)
        A face_profile_index data array contains the indexs of all the face profile names

    """
    delete_empty_profile(face_profile_directory)  # delete profile directory without images

    # Get a the list of folder names in face_profile as the profile names
    face_profile_names = [d for d in os.listdir(face_profile_directory) if "." not in str(d)]

    if len(face_profile_names) < 2: 
        logging.error("\nFace profile contains too little profiles (At least 2 profiles are needed)")
        exit()

    first_data = str(face_profile_names[0])
    first_data_path = os.path.join(face_profile_directory, first_data)
    X1, y1 = read_images_from_single_face_profile(first_data_path, 0)
    X_data = X1   
    Y_data = y1   

    print "Loading Database: "
    print 0,"    ", first_data_path
    for i in range(0, len(face_profile_names)):
        directory_name = str(face_profile_names[i])
        directory_path = os.path.join(face_profile_directory, directory_name)
        tempX, tempY = read_images_from_single_face_profile(directory_path, i)
        X_data = np.concatenate((X_data, tempX), axis=0)
        Y_data = np.append(Y_data, tempY)

        print i, "    ",tempX.shape[0]," images are loaded from:", directory_path

    return X_data, Y_data, face_profile_names


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

