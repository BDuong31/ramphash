import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report # <--- THÊM: Xem chi tiết độ chính xác từng class

# 1. Load & Split (Cố định random_state để kết quả tái tạo chính xác ở mọi lần chạy)
X, y = np.load('X_data.npy'), np.load('y_labels.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Khởi tạo RF với các tham số tối ưu hơn cho bài toán phân loại tư thế
# n_jobs=-1 giúp tận dụng TẤT CẢ lõi CPU của máy để train nhanh hơn gấp nhiều lần
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 3. Đánh giá mô hình chuyên sâu
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"=== ĐỘ CHÍNH XÁC RANDOM FOREST: {acc * 100:.2f}% ===")

LABELS = ['AHEAD', 'RIGHT', 'LEFT', 'STOP', 'NONE']
print("\nBáo cáo chi tiết theo từng hành động:")
print(classification_report(y_test, y_pred, target_names=LABELS))

# 4. Lưu mô hình
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/marshaller_model_rf.pkl')
print("Đã huấn luyện và lưu RF thành công.")