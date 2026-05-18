import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load dữ liệu và mô hình chuẩn
X = np.load('X_data.npy')
y = np.load('y_labels.npy')
rf_model = joblib.load('models/marshaller_model_rf.pkl') # Sửa đường dẫn

labels = ['ahead', 'right', 'left', 'stop', 'none']

# 2. Dự đoán
y_pred = rf_model.predict(X)

# 3. In báo cáo chi tiết
print("--- CHI TIẾT ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST ---")
print(classification_report(y, y_pred, target_names=labels))

# 4. Vẽ Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Actual)')
plt.show()

# 5. TỐI ƯU: Giải mã Feature Importance thành tên khớp xương MediaPipe
importances = rf_model.feature_importances_

# Danh sách tên các mốc quan trọng thường ảnh hưởng đến hành động điều hướng (Marshaller)
MP_POSE_LANDMARKS = {
    11: "LEFT_SHOULDER", 12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",    16: "RIGHT_WRIST",
    19: "LEFT_INDEX",    20: "RIGHT_INDEX"
}
# Các chỉ số tương ứng với tọa độ
COORD_MAPPING = {0: 'x', 1: 'y', 2: 'z', 3: 'visibility'}

print("\n--- TOP 5 ĐẶC TRƯNG QUYẾT ĐỊNH ĐẾN KẾT QUẢ CỦA RF ---")
indices = np.argsort(importances)[-5:][::-1]

for idx in indices:
    landmark_id = idx // 4  # Tìm xem thuộc landmark số mấy (0-32)
    coord_id = idx % 4      # Tìm xem là trục x, y, z hay độ hiển thị v
    
    # Lấy tên khớp xương nếu nằm trong danh sách thường dùng, ngược lại hiện ID gốc
    landmark_name = MP_POSE_LANDMARKS.get(landmark_id, f"LANDMARK_{landmark_id}")
    coord_name = COORD_MAPPING[coord_id]
    
    print(f"Vị trí {landmark_name} (Trục {coord_name}) | Trọng số ảnh hưởng: {importances[idx]:.4f}")