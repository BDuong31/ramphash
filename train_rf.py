import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Định nghĩa thư mục đầu vào cho dữ liệu huấn luyện, lưu trữ mô hình và biểu đồ đánh giá
DATA_DIR = "processed_npy_data"
MODEL_DIR = "models"
RESULTS_DIR = "results/rf"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) # Tạo thư mục lưu kết quả

# Tải các ma trận dữ liệu đã được tiền xử lý
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
classes = np.load(os.path.join(DATA_DIR, 'classes.npy'), allow_pickle=True)

print("Huấn luyện Random Forest vững chắc (24 đặc trưng đầu vào)...")
# Khởi tạo Random Forest với 250 cây quyết định, trọng số class cân bằng và song song hóa đa luồng
rf_model = RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Dự đoán trên tập dữ liệu kiểm tra
y_pred = rf_model.predict(X_test)
print(f"\n[RANDOM FOREST ACCURACY]: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Tạo báo cáo chi tiết các độ đo phân loại (precision, recall, f1-score)
report = classification_report(y_test, y_pred, target_names=classes)
print(report)

# Đóng gói mô hình Random Forest đã huấn luyện xuống đĩa
joblib.dump(rf_model, os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
print("Đã đóng gói thành công file mô hình: models/random_forest_model.pkl")

# Tạo và lưu heatmap cho Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
plt.close()

# Lưu báo cáo phân loại vào tệp văn bản
with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

# Tính toán và vẽ biểu đồ độ quan trọng tương đối của các đặc trưng (cho 24 đặc trưng đã chuẩn hóa)
importances = rf_model.feature_importances_
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances, color='skyblue', edgecolor='black')
plt.title('Random Forest Feature Importances (24 Features)')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.xticks(range(len(importances))) # Hiển thị từng số thứ tự đặc trưng
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))
plt.close()

print(f"Đã xuất toàn bộ biểu đồ và báo cáo của Random Forest ra thư mục: {RESULTS_DIR}")