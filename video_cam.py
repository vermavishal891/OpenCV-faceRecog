import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

base_frame = None

while True:
    check, img = video.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    current_frame = cv2.GaussianBlur(gray,(21,21),0)

    faces = face_cascade.detectMultiScale(current_frame,1.3,5)

    for (x,y,w,h) in faces:
    	cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("IMAGE",img)

    key=cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows
