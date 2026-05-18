import numpy as np
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # <--- THÊM: Chuẩn hóa dữ liệu
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # <--- THÊM: Batch Normalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping # <--- THÊM: Early Stopping

# 1. Load & Split
X, y = np.load('X_data.npy'), np.load('y_labels.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. CHUẨN HÓA DỮ LIỆU (Giúp DNN học nhanh và ổn định hơn rất nhiều)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lưu lại scaler để khi deploy bên file API/Socket cũng phải dùng scaler này transform ảnh đầu vào
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/marshaller_scaler.pkl')

y_train_oh = to_categorical(y_train, 5)
y_test_oh = to_categorical(y_test, 5)

# 3. Callbacks nâng cao
log_dir = os.path.join("logs", "dnn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Tự động dừng train nếu loss trên tập test không giảm sau 10 epochs (Chống Overfitting)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 4. Kiến trúc mạng tối ưu hơn (Thêm Batch Normalization để tăng tốc độ hội tụ)
model = Sequential([
    Dense(128, activation='relu', input_shape=(132,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Đang huấn luyện DNN với cơ chế tối ưu...")
model.fit(
    X_train, y_train_oh, 
    epochs=150, # Nâng tối đa lên 150 nhưng có Early Stopping lo việc dừng sớm
    batch_size=32, 
    validation_data=(X_test, y_test_oh), 
    callbacks=[tb_callback, early_stop], 
    verbose=1
)

model.save('models/marshaller_model_dnn.h5')
print("Đã lưu mô hình DNN tối ưu.")