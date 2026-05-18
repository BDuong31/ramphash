import numpy as np
import datetime
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

# Load & Split
X, y = np.load('X_data.npy'), np.load('y_labels.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
y_train_oh = to_categorical(y_train, 5)
y_test_oh = to_categorical(y_test, 5)

# TensorBoard log
log_dir = os.path.join("logs", "dnn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model = Sequential([
    Dense(64, activation='relu', input_shape=(132,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Đang huấn luyện DNN...")
model.fit(X_train, y_train_oh, epochs=100, validation_data=(X_test, y_test_oh), callbacks=[tb_callback], verbose=1)
model.save('models/marshaller_model_dnn.h5')