import numpy as np
import io
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    img = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(img, 0)

X, y = np.load('X_data.npy'), np.load('y_labels.npy')
dnn = load_model('models/marshaller_model_dnn.h5')
rf = joblib.load('models/marshaller_model_rf.pkl')
labels = ['ahead', 'right', 'left', 'stop', 'none']

# Predict
y_dnn = np.argmax(dnn.predict(X), axis=1)
y_rf = rf.predict(X)

# Ghi hình ảnh vào TensorBoard
writer = tf.summary.create_file_writer("logs/comparison")

for name, preds in [("DNN", y_dnn), ("RF", y_rf)]:
    cm = confusion_matrix(y, preds)
    fig = plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
    plt.title(f"Confusion Matrix: {name}")
    
    with writer.as_default():
        tf.summary.image(f"ConfusionMatrix_{name}", plot_to_image(fig), step=0)

print("Đã xuất so sánh vào TensorBoard tab IMAGES.")