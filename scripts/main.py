""" 
====================================================
    Faces recognition and detection using OpenCV 
====================================================

Summary:
        Facial Recognition Using OpenCV and SVM

        Created by:  Chenxing Ouyang

"""
import cv2
import numpy as np
from scipy import ndimage

import utils as ut

print(__doc__)


# dictionary mapping used to keep track of head rotation maps
rotation_maps = {
    "left": np.array([-30, 0, 30]),
    "right": np.array([30, 0, -30]),
    "middle": np.array([0, -30, 30]),
}

def get_rotation_map(angle):
    if angle > 0: return rotation_maps.get("right", None)
    if angle < 0: return rotation_maps.get("left", None)
    if angle == 0: return rotation_maps.get("middle", None)


print get_rotation_map(0)