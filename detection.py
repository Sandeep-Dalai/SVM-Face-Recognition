import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path + "/trainer/trainer.yml")
cascadePath = path + "/Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)

        if nbr_predicted == 1:
            nbr_predicted = 'Bhargavi'
        elif nbr_predicted == 2:
            nbr_predicted = 'Sandeep'
        elif nbr_predicted == 3:
            nbr_predicted = 'Aishwarya'
        elif nbr_predicted == 4:
            nbr_predicted = 'Narendra Modi'
        elif nbr_predicted == 5:
            nbr_predicted = 'Virat Kohli'
        
        cv2.putText(im, str(nbr_predicted),(x, y+h), font, 1.1, (0, 255, 0), 2)

    cv2.imshow('im', im)
    key = cv2.waitKey(1)
    if key == 27:  
        break

cam.release()
cv2.destroyAllWindows()
