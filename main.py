import cv2
import sys
import winsound
import os
print(cv2.__file__)
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'haarcascade_frontalface_default.xml')

faceCascade = cv2.CascadeClassifier(filename)

video_capture = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) == 0:
        break
    #if len(faces) != 0:
        #winsound.Beep(frequency=500, duration=300)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()