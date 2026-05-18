import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def run_check(base_path):
    categories = ['ahead', 'right', 'left', 'stop', 'none']
    for category in categories:
        path = os.path.join(base_path, category)
        if not os.path.exists(path): continue
        print(f"Checking: {category}")
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name))
            if img is None: continue
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Check Data', img)
            if cv2.waitKey(1) & 0xFF == 27: break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_check('data_marshaller_final')