import cv2
import mediapipe as mp
import os
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def create_data(base_path):
    X, y = [], []
    label_map = {'ahead': 0, 'right': 1, 'left': 2, 'stop': 3, 'none': 4}
    
    for category, label_id in label_map.items():
        folder = os.path.join(base_path, category)
        if not os.path.exists(folder): continue
        print(f"Processing {category}...")
        for img_name in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_name))
            if img is None: continue
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                # Lấy 33 điểm * (x, y, z, visibility) = 132 features
                lms = [ [lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
                X.append(np.array(lms).flatten())
                y.append(label_id)
                
    np.save('X_data.npy', np.array(X))
    np.save('y_labels.npy', np.array(y))
    print(f"Saved! Shape: {np.array(X).shape}")

if __name__ == "__main__":
    create_data('data_marshaller_final')