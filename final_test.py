import numpy as np
import io
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score # <--- THÊM: Tính thêm Acc để log
import seaborn as sns

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    img = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(img, 0)

# 1. Load dữ liệu và các mô hình
X, y = np.load('X_data.npy'), np.load('y_labels.npy')
dnn = load_model('models/marshaller_model_dnn.h5')
rf = joblib.load('models/marshaller_model_rf.pkl')
labels = ['ahead', 'right', 'left', 'stop', 'none']

# 2. XỬ LÝ CHUẨN HÓA DỮ LIỆU RIÊNG CHO DNN
if os.path.exists('models/marshaller_scaler.pkl'):
    scaler = joblib.load('models/marshaller_scaler.pkl')
    X_dnn_scaled = scaler.transform(X) # Chuẩn hóa ma trận X dành riêng cho DNN
else:
    print("Cảnh báo: Không tìm thấy file 'marshaller_scaler.pkl'. Kết quả DNN có thể bị sai!")
    X_dnn_scaled = X

# 3. Dự đoán (Thêm verbose=0 cho DNN để sạch terminal)
y_dnn = np.argmax(dnn.predict(X_dnn_scaled, verbose=0), axis=1)
y_rf = rf.predict(X)

# 4. Ghi kết quả vào TensorBoard
writer = tf.summary.create_file_writer("logs/comparison")

for name, preds in [("DNN", y_dnn), ("RF", y_rf)]:
    # Tính toán Ma trận nhầm lẫn
    cm = confusion_matrix(y, preds)
    acc = accuracy_score(y, preds) # Tính thêm chỉ số Accuracy tổng quan
    
    # Vẽ đồ thị bằng Matplotlib & Seaborn
    fig, ax = plt.subplots(figsize=(6, 6))
    # Đổi bảng màu sang 'Blues' cho RF và 'Purples' cho DNN để phân biệt rõ ràng khi xem trên web
    cmap_color = 'Purples' if name == "DNN" else 'Blues'
    
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap=cmap_color, ax=ax)
    ax.set_title(f"Confusion Matrix: {name} (Acc: {acc*100:.2f}%)")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Ghi ảnh đồ thị vào TensorBoard
    with writer.as_default():
        tf.summary.image(f"ConfusionMatrix_{name}", plot_to_image(fig), step=0)
        # Gợi ý: Ghi thêm cả điểm số dạng Text/Scalar để dễ so sánh ở tab SCALARS
        tf.summary.scalar(f"Accuracy_Overall/{name}", acc, step=0)

print("--- HOÀN THÀNH ---")
print("Đã xuất so sánh cấu hình tối ưu vào TensorBoard.")
print("Bật terminal gõ: tensorboard --logdir=logs để kiểm tra kết quả tại tab IMAGES và SCALARS.")