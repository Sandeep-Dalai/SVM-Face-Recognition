import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

path = os.path.dirname(os.path.abspath(__file__))

face_cascade_path = path + "/Classifiers/face.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path + "/trainer/trainer.yml")


mask_detector = load_model('mask_detector.model')


GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


cam = cv2.VideoCapture(0)


while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        
        nbr_predicted, conf = recognizer.predict(roi_gray)
        cv2.rectangle(frame, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)

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

        
        face = cv2.resize(roi_color, (224, 224))
        face = face.astype("float") / 255.0
        pred = mask_detector.predict(np.expand_dims(face, axis=0))[0][0]

        
        color = GREEN if pred > 0.8 else RED
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = "Mask" if pred > 0.8 else "No Mask"
        cv2.putText(frame, f'{nbr_predicted} - {label}', (x, y-10), FONT, 1.1, color, 2)

    cv2.imshow('Face Recognition with Mask Detection', frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
