"""
Real time facial tracking and recognition using harr cascade and neural network

"""

import cv2
import numpy as np
from math import sin, cos, radians
from scipy import ndimage


ANGLES_LEFT = np.array([-30, 0, 30])
ANGLES_RIGHT = np.array([30, 0, -30])
ANGLES_MIDDLE = np.array([0, -30, 30])
angles = ANGLES_MIDDLE 
SKIP_FRAME = 2
frame_skip_rate = 0 # skip 1 frame every other frame
SCALE_FACTOR = 4 # used to resize the captured frame for face detection for faster processing speed
face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml") #create a cascade classifier
sideFace_cascade = cv2.CascadeClassifier('../data/haarcascade_profileface.xml')


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


webcam = cv2.VideoCapture(0)
cv2.namedWindow("Real Time Facial Recognition")

ret, frame = webcam.read() # get first frame
frame_scale = (frame.shape[1]/SCALE_FACTOR,frame.shape[0]/SCALE_FACTOR)  # (y, x)

while ret:

    # resize the captured frame for face detection to increase processing speed

    resized_frame = cv2.resize(frame, frame_scale)

    processed_frame = resized_frame

    # Skip a frame if the no face was found last frame
    if frame_skip_rate == 0:
        faceFound = False
        for angle in angles:
            # rotated_frame = rotate_image(resized_frame, angle)

            rotated_frame = ndimage.rotate(resized_frame, angle)

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
            faces = faces if len(faces) else sideFace_cascade.detectMultiScale(gray)

            # for f in faces:
            #     x, y, w, h = [ v*SCALE_FACTOR for v in f ] # scale the bounding box back to original frame size
            #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
            #     cv2.putText(frame, "DumbAss", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

            if len(faces):
                for f in faces:
                    x, y, w, h = [ v for v in f ] # scale the bounding box back to original frame size
                    cv2.rectangle(rotated_frame, (x,y), (x+w,y+h), (0,255,0))
                    cv2.putText(rotated_frame, "DumbAss", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

                # rotate the frame back and trim the black paddings
                processed_frame = trim(rotate_image(rotated_frame, angle * (-1)), frame_scale)

                # reset the optmized angles array
                if angle > 0: angles = ANGLES_RIGHT
                if angle < 0: angles = ANGLES_LEFT
                if angle == 0: angles = ANGLES_MIDDLE

                faceFound = True

                break

        if faceFound: 
            frame_skip_rate = 0
        else:
            frame_skip_rate = SKIP_FRAME
            print "face not found"

    else:
        frame_skip_rate -= 1
        print "face not found"


  
    cv2.putText(processed_frame, "Press ESC or 'q' to quit.", (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

    cv2.imshow("Real Time Facial Recognition", processed_frame)

    # rotate ANGLES to adapt the camera to the user rotated angle
    # ANGLES = np.roll(ANGLES, np.where(ANGLES==angle)[0])


    # get next frame
    ret, frame = webcam.read()

    key = cv2.waitKey(1)
    # exit on 'q' 'esc' 'Q'
    if key in [27, ord('Q'), ord('q')]: 
        break

webcam.release()
cv2.destroyAllWindows()