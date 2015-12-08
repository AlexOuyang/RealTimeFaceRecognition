import cv2
import sys

SCALE_FACTOR = 4
# cascPath = sys.argv[1]
# face_cascade = cv2.CascadeClassifier(cascPath)
# face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('../data/haarcascade_profileface.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()


    frame = cv2.resize(frame, (frame.shape[1]/SCALE_FACTOR,frame.shape[0]/SCALE_FACTOR))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    key = cv2.waitKey(1)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()