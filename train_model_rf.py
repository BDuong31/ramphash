import numpy as np
import joblib
import os
import datetime
from sklearn.model_selection import train_test_split # <--- THÊM DÒNG NÀY
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf

X, y = np.load('X_data.npy'), np.load('y_labels.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Log to TensorBoard
log_dir = os.path.join("logs", "rf_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    tf.summary.scalar('Accuracy/Test', accuracy_score(y_test, rf.predict(X_test)), step=1)

joblib.dump(rf, 'models/marshaller_model_rf.pkl')
print("Đã huấn luyện và lưu RF.")