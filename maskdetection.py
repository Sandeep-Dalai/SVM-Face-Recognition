import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

path = os.path.dirname(os.path.abspath(__file__))


cascadePath = path + "/Classifiers/face.xml"
face_detector = cv2.CascadeClassifier(cascadePath)


mask_detector = load_model('mask_detector.model')


GREEN = (0, 255, 0)
RED = (0, 0, 255)


cap = cv2.VideoCapture(0)


while True:
   
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    
    for (x, y, w, h) in faces:
        
        face = frame[y:y+h, x:x+w]

        
        face = cv2.resize(face, (224, 224))

        
        face = face.astype("float") / 255.0

       
        pred = mask_detector.predict(np.expand_dims(face, axis=0))[0][0]

        
        color = GREEN if pred > 0.5 else RED
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        
        label = "Mask" if pred > 0.5 else "No Mask"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    cv2.imshow('Face Mask Detection', frame)

    
    key = cv2.waitKey(1)
    if key == 27: 
        break


cap.release()
cv2.destroyAllWindows()
