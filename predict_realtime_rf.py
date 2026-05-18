import cv2
import mediapipe as mp
import numpy as np
import joblib

# Khởi tạo MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load mô hình Random Forest
rf_model = joblib.load('marshaller_rf_model.pkl')
labels = ['AHEAD', 'RIGHT', 'LEFT', 'STOP', 'NONE']

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    
    if res.pose_landmarks:
        # Trích xuất tọa độ y hệt lúc train
        lms = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
        input_data = np.array(lms).flatten().reshape(1, -1)
        
        # Dự đoán nhãn và xác suất
        class_id = rf_model.predict(input_data)[0]
        probs = rf_model.predict_proba(input_data)[0]
        confidence = probs[class_id]
        
        label_text = f"RF: {labels[class_id]} ({confidence:.2f})"
        
        # Vẽ lên màn hình
        cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Realtime Marshaller - Random Forest', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()