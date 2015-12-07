import cv2
import numpy as np
from scipy import ndimage


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

