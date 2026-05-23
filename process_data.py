import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Khởi tạo bộ ước lượng MediaPipe Pose
mp_pose = mp.solutions.pose
# Thiết lập mô hình Pose: static_image_mode=True được tối ưu hóa cho các ảnh tĩnh riêng lẻ
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
 
# Thư mục chứa ảnh gốc đầu vào và nơi lưu các tệp CSV kết quả
DATA_DIR = "data_marshaller_final" 
OUTPUT_DIR = "output_csv"         
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách các tọa độ 2D (X, Y) được trích xuất từ các khớp cơ thể, kết thúc bằng tọa độ cổ được tính toán
COLUMNS = [
    'noseX', 'noseY', 'left_eyeX', 'left_eyeY', 'right_eyeX', 'right_eyeY',
    'left_earX', 'left_earY', 'right_earX', 'right_earY',
    'left_shoulderX', 'left_shoulderY', 'right_shoulderX', 'right_shoulderY',
    'left_elbowX', 'left_elbowY', 'right_elbowX', 'right_elbowY',
    'left_wristX', 'left_wristY', 'right_wristX', 'right_wristY',
    'left_hipX', 'left_hipY', 'right_hipX', 'right_hipY',
    'left_kneeX', 'left_kneeY', 'right_kneeX', 'right_kneeY',
    'left_ankleX', 'left_ankleY', 'right_ankleX', 'right_ankleY',
    'neckX', 'neckY'
]

def extract_keypoints_from_folder(folder_path):
    """Duyệt qua các ảnh trong thư mục và trích xuất tọa độ khung xương."""
    folder_data = []
    
    # Các định dạng ảnh tĩnh được hỗ trợ
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)])
    
    # Xử lý các ảnh với thanh tiến trình theo dõi
    for img_name in tqdm(images, desc=f"Processing {os.path.basename(folder_path)}"):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Chuyển đổi BGR (định dạng OpenCV) sang RGB (định dạng MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Khởi tạo các đặc trưng mặc định là 0.0 đề phòng trường hợp phát hiện thất bại
        row_data = {col: 0.0 for col in COLUMNS}
        
        # Trích xuất vị trí các điểm mốc nếu phát hiện thấy tư thế dáng người
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Ánh xạ các điểm mốc vùng đầu
            row_data['noseX'], row_data['noseY'] = landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y
            row_data['left_eyeX'], row_data['left_eyeY'] = landmarks[mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE].y
            row_data['right_eyeX'], row_data['right_eyeY'] = landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y
            row_data['left_earX'], row_data['left_earY'] = landmarks[mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR].y
            row_data['right_earX'], row_data['right_earY'] = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y
            
            # Ánh xạ các điểm mốc thân trên (vai, khuỷu tay, cổ tay)
            row_data['left_shoulderX'], row_data['left_shoulderY'] = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            row_data['right_shoulderX'], row_data['right_shoulderY'] = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            row_data['left_elbowX'], row_data['left_elbowY'] = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
            row_data['right_elbowX'], row_data['right_elbowY'] = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            row_data['left_wristX'], row_data['left_wristY'] = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            row_data['right_wristX'], row_data['right_wristY'] = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            
            # Ánh xạ các điểm mốc thân dưới (hông, đầu gối, mắt cá chân)
            row_data['left_hipX'], row_data['left_hipY'] = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            row_data['right_hipX'], row_data['right_hipY'] = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            row_data['left_kneeX'], row_data['left_kneeY'] = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
            row_data['right_kneeX'], row_data['right_kneeY'] = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
            row_data['left_ankleX'], row_data['left_ankleY'] = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
            row_data['right_ankleX'], row_data['right_ankleY'] = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            
            # Tính toán tọa độ Cổ là điểm trung bình giữa vai trái và vai phải
            row_data['neckX'] = (row_data['left_shoulderX'] + row_data['right_shoulderX']) / 2
            row_data['neckY'] = (row_data['left_shoulderY'] + row_data['right_shoulderY']) / 2

        folder_data.append(row_data)
        
    return folder_data

# Duyệt qua tất cả các thư mục nhãn bên trong thư mục ảnh thô
for folder_name in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder_name)
    
    if os.path.isdir(folder_path):
        keypoints_list = extract_keypoints_from_folder(folder_path)
        
        # Lưu ma trận đặc trưng vào file CSV nếu trích xuất thành công
        if keypoints_list:
            df = pd.DataFrame(keypoints_list, columns=COLUMNS)
            output_file_path = os.path.join(OUTPUT_DIR, f"{folder_name}.csv")
            df.to_csv(output_file_path, index=False)
            print("Saved: ", output_file_path, "with", len(df), "rows.")

# Giải phóng tài nguyên MediaPipe
pose.close()
print("Hoàn thành trích xuất dữ liệu tất cả các thư mục!")