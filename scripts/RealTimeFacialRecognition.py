"""
Real time facial tracking and recognition using harr cascade and neural network

"""

import cv2

SCALE_FACTOR = 4 # used to resize the captured frame for face detection for faster processing speed
face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml")  # create a cascade classifier 
webface_cascade = cv2.VideoCapture(0)
cv2.namedWindow("Real Time Facial Recognition")

ret, frame = webface_cascade.read() # get first frame

while ret:

    # resize the captured frame for face detection to increase processing speed
    resized_frame = cv2.resize(frame, (frame.shape[1]/SCALE_FACTOR,frame.shape[0]/SCALE_FACTOR))

    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

    for f in faces:
        x, y, w, h = [ v*SCALE_FACTOR for v in f ] # scale the bounding box back to original frame size
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
        cv2.putText(frame, "DumbAss", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

    cv2.putText(frame, "Press ESC or 'q' to quit.", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

    cv2.imshow("Real Time Facial Recognition", frame)

    # get next frame
    ret, frame = webface_cascade.read()

    key = cv2.waitKey(1)
    # exit on 'q' 'esc' 'Q'
    if key in [27, ord('Q'), ord('q')]: 
        break

webface_cascade.release()
cv2.destroyAllWindows()