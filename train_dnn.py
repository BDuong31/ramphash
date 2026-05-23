import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Định nghĩa thư mục lưu trữ mô hình và log
DATA_DIR = "processed_npy_data"
MODEL_DIR = "models"
RESULTS_DIR = "results/dnn"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) # Tạo thư mục lưu kết quả

# Tải tập dữ liệu đã chuẩn hóa
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
classes = np.load(os.path.join(DATA_DIR, 'classes.npy'), allow_pickle=True)

# Xây dựng kiến trúc mạng nơ-ron sâu tuần tự
model = Sequential([
    # Lớp ẩn đầu tiên với 128 nút và hàm kích hoạt ReLU (24 đặc trưng đầu vào)
    Dense(128, activation='relu', input_shape=(24,)), 
    BatchNormalization(),  # Chuẩn hóa kích thước kích hoạt
    Dropout(0.4),          # Giảm thiểu quá khớp bằng cách bỏ 40% số nút
    
    # Lớp ẩn thứ hai
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),          # Bỏ 30% số nút
    
    # Lớp ẩn thứ ba và lớp đầu ra softmax
    Dense(32, activation='relu'),
    Dense(len(classes), activation='softmax')  # Dự đoán xác suất cho từng tín hiệu tay
])

# Biên dịch mô hình sử dụng thuật toán Adam và loss Sparse Categorical Crossentropy (do nhãn là chỉ số nguyên)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Thiết lập dừng sớm (early stopping): dừng khớp nếu val_loss không cải thiện sau 15 epoch liên tiếp
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
csv_logger = CSVLogger(os.path.join(RESULTS_DIR, 'training_log.csv'))

print("Huấn luyện mạng Deep Neural Network (24 đặc trưng đầu vào)...")
history = model.fit(
    X_train, y_train, 
    epochs=150, batch_size=32, 
    validation_data=(X_test, y_test), 
    callbacks=[early_stop, csv_logger], 
    verbose=1
)

# Đánh giá hiệu suất trên tập kiểm thử
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[DNN ACCURACY]: {acc * 100:.2f}%")

# Lưu trọng số và cấu hình mô hình
model.save(os.path.join(MODEL_DIR, 'dnn_pose_model.keras'))
print("Đã lưu thành công file mô hình gốc: models/dnn_pose_model.keras")

# Vẽ biểu đồ tiến trình độ chính xác (Accuracy) huấn luyện và kiểm thử
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('DNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_plot.png'))
plt.close()

# Vẽ biểu đồ tiến trình độ mất mát (Loss) huấn luyện và kiểm thử
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('DNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_plot.png'))
plt.close()

# Lấy dự đoán của mô hình trên tập kiểm thử
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Tạo và lưu heatmap cho Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('DNN Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
plt.close()

# Lưu báo cáo phân loại dạng văn bản
report = classification_report(y_test, y_pred, target_names=classes)
with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Đã xuất toàn bộ biểu đồ và log của DNN ra thư mục: {RESULTS_DIR}")