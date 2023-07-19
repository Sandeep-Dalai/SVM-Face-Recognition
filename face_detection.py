import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))
face_cap=cv2.CascadeClassifier(path+r'\Classifiers\face.xml')
video_cap=cv2.VideoCapture(0)
while True:
    ret,video_data=video_cap.read()
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",video_data)
    key = cv2.waitKey(1)
    if key == 27:  # Press Esc to exit
        break
video_cap.release()