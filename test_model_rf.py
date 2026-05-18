import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load dữ liệu và mô hình
X = np.load('X_data.npy')
y = np.load('y_labels.npy')
rf_model = joblib.load('marshaller_rf_model.pkl')

labels = ['ahead', 'right', 'left', 'stop', 'none']

# 2. Dự đoán trên toàn bộ tập dữ liệu (hoặc bạn có thể chia tập test riêng)
y_pred = rf_model.predict(X)

# 3. In báo cáo chi tiết (Precision, Recall, F1-score)
print("--- CHI TIẾT ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST ---")
print(classification_report(y, y_pred, target_names=labels))

# 4. Vẽ Confusion Matrix (Ma trận nhầm lẫn)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Actual)')
plt.show()

# 5. Kiểm tra mức độ quan trọng của các điểm mốc (Feature Importance)
importances = rf_model.feature_importances_
# Vì mỗi điểm có 4 giá trị (x, y, z, v), ta có thể gộp lại để xem điểm nào quan trọng nhất
print("\nTop 5 đặc trưng quan trọng nhất (theo chỉ số feature):")
indices = np.argsort(importances)[-5:][::-1]
for i in indices:
    print(f"Feature index {i}: {importances[i]:.4f}")