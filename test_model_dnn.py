import numpy as np
import joblib  # <--- THÊM: Để load scaler
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import os

# 1. Load dữ liệu và mô hình chuẩn
X = np.load('X_data.npy')
y = np.load('y_labels.npy')

# Cập nhật đường dẫn load mô hình chính xác từ thư mục models
model = load_model('models/marshaller_model_dnn.h5')

# BẮT BUỘC: Load scaler đã lưu lúc train để transform dữ liệu test
if os.path.exists('models/marshaller_scaler.pkl'):
    scaler = joblib.load('models/marshaller_scaler.pkl')
    X_scaled = scaler.transform(X)
else:
    print("Cảnh báo: Không tìm thấy file scaler, kết quả predict có thể bị sai lệch!")
    X_scaled = X

# 2. Dự đoán
y_pred = np.argmax(model.predict(X_scaled, verbose=0), axis=1)
labels = ['ahead', 'right', 'left', 'stop', 'none']

# 3. Báo cáo đánh giá
print("--- CHI TIẾT ĐÁNH GIÁ MÔ HÌNH DNN ---")
print(classification_report(y, y_pred, target_names=labels))

# 4. Vẽ Confusion Matrix (Bổ sung cmap để dễ nhìn giống bên RF)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Purples')
plt.title('Confusion Matrix - DNN')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Actual)')

# --- LOG COFUSION MATRIX LÊN TENSORBOARD (Tận dụng hàm bạn viết) ---
def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

# Tạo log TensorBoard để bạn có thể xem biểu đồ ma trận nhầm lẫn từ xa qua web
log_dir = os.path.join("logs", "test_dnn_cm")
file_writer_cm = tf.summary.create_file_writer(log_dir)

# Tạo lại một bản figure riêng để ghi log tránh bị hàm plt.show() giải phóng trước
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Purples', ax=ax)
ax.set_title('Confusion Matrix - DNN')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", plot_to_image(fig), step=0)

plt.show() # Hiển thị lên màn hình local/notebook