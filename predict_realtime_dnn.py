import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
model = load_model('marshaller_model.h5')
labels = ['AHEAD', 'RIGHT', 'LEFT', 'STOP', 'NONE']

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    
    if res.pose_landmarks:
        lms = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
        input_data = np.array(lms).flatten().reshape(1, 132)
        
        prediction = model.predict(input_data, verbose=0)
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]
        
        label = f"{labels[class_id]} ({confidence:.2f})"
        color = (0, 255, 0) if labels[class_id] != 'STOP' else (0, 0, 255)
        
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Realtime Marshaller Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()