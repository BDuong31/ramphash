import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "Dataset"
OUTPUT_DIR = "processed_npy_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Các cột được chọn cho mô hình thân trên: tổng cộng 24 đặc trưng (tọa độ).
# Lưu ý: bao gồm các mục trùng lặp cho cổ tay để khớp với yêu cầu huấn luyện cụ thể.
COLUMNS_24 = [
    'noseX', 'noseY', 'left_eyeX', 'left_eyeY', 'right_eyeX', 'right_eyeY',
    'left_earX', 'left_earY', 'right_earX', 'right_earY',
    'left_shoulderX', 'left_shoulderY', 'right_shoulderX', 'right_shoulderY',
    'left_elbowX', 'left_elbowY', 'right_elbowX', 'right_elbowY',
    'left_wristX', 'left_wristY', 'right_wristX', 'right_wristY',
    'right_wristX_dup', 'right_wristY_dup',
    'neckX', 'neckY'
]

# Tên các điểm khớp thực tế tương ứng với chỉ số trong mảng đầu ra (24 đặc trưng, không trùng lặp)
ACTUAL_COLUMNS = [
    'noseX', 'noseY', 'left_eyeX', 'left_eyeY', 'right_eyeX', 'right_eyeY',
    'left_earX', 'left_earY', 'right_earX', 'right_earY',
    'left_shoulderX', 'left_shoulderY', 'right_shoulderX', 'right_shoulderY',
    'left_elbowX', 'left_elbowY', 'right_elbowX', 'right_elbowY',
    'left_wristX', 'left_wristY', 'right_wristX', 'right_wristY',
    'neckX', 'neckY'
]

def normalize_upper_body_features(features_2d):
    """
    Chuẩn hóa hình học 24 đặc trưng thân trên.
    Tịnh tiến tọa độ lấy Cổ (index 22, 23) làm tâm gốc (0,0), scale chia theo độ rộng vai.
    """
    norm_features = features_2d.copy()
    for i in range(len(norm_features)):
        row = norm_features[i]
        neck_x, neck_y = row[22], row[23]
        
        # Tránh chia cho 0 hoặc tịnh tiến các tọa độ rỗng
        if neck_x == 0 and neck_y == 0:
            continue
            
        # Tọa độ Vai trái (index 10, 11), Vai phải (index 12, 13)
        ls_x, ls_y = row[10], row[11]
        rs_x, rs_y = row[12], row[13]
        
        # Tính khoảng cách Euclid giữa hai vai (hệ số scale)
        shoulder_dist = np.sqrt((ls_x - rs_x)**2 + (ls_y - rs_y)**2)
        
        if shoulder_dist == 0:
            shoulder_dist = 1.0
            
        # Tịnh tiến gốc tọa độ về Cổ và chia tỷ lệ theo khoảng cách vai
        for j in range(0, 24):
            if j % 2 == 0:  # Trục X
                row[j] = (row[j] - neck_x) / shoulder_dist
            else:          # Trục Y
                row[j] = (row[j] - neck_y) / shoulder_dist
                
        norm_features[i] = row
    return norm_features

X_all = []
y_all = []

# Lấy danh sách tất cả các file CSV trong thư mục dataset đầu vào
csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
print(f"Đang trích xuất ma trận dữ liệu thân trên từ {DATASET_DIR}...")

for file_name in csv_files:
    # Dùng tên file viết hoa (không có đuôi .csv) làm nhãn cử chỉ (ví dụ: AHEAD, LEFT, RIGHT, STOP)
    label_str = file_name.replace('.csv', '').upper()
    file_path = os.path.join(DATASET_DIR, file_name)
    
    df = pd.read_csv(file_path)
    if df.empty:
        continue
        
    # Chỉ trích xuất 24 đặc trưng thân trên: 22 đặc trưng đầu tiên (từ mũi đến cổ tay phải) kết hợp với tọa độ cổ (từ chỉ số 34:36 trong cấu trúc bảng gốc)
    raw_upper_body = np.hstack((df.iloc[:, :22].values, df.iloc[:, 34:36].values))
    
    X_all.append(raw_upper_body)
    y_all.extend([label_str] * len(raw_upper_body))
    print(f"  |-- Khớp nhãn [{label_str}] trích xuất thành công: {len(raw_upper_body)} dòng dữ liệu.")

# Xếp chồng dữ liệu thành các ma trận NumPy
X_all = np.vstack(X_all)
y_all = np.array(y_all)

# Chuyển đổi các lớp chuỗi (ví dụ: AHEAD) thành số nguyên (0, 1, 2...) cho các thuật toán học máy
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_all)
np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), label_encoder.classes_)

print("\nĐang thực thi tính toán tối ưu chuẩn hóa ma trận hình học tương đối...")
X_all_normalized = normalize_upper_body_features(X_all)

# Chia dữ liệu: 80% huấn luyện, 20% kiểm tra. Sử dụng phân tầng để cân bằng phân phối lớp.
X_train, X_test, y_train, y_test = train_test_split(
    X_all_normalized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Lưu trữ tập dữ liệu đã xử lý dưới dạng file nhị phân numpy cho các script huấn luyện mô hình
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

print(f"Hoàn tất quy trình làm sạch dữ liệu gốc! Thứ tự danh sách nhãn: {label_encoder.classes_.tolist()}")