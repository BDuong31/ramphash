import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
import tensorflow as tf

def plot_to_image(figure):
    """Chuyển đổi Matplotlib figure thành hình ảnh để TensorBoard có thể đọc"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


X = np.load('X_data.npy')
y = np.load('y_labels.npy')
model = load_model('marshaller_model.h5')

y_pred = np.argmax(model.predict(X), axis=1)
labels = ['ahead', 'right', 'left', 'stop', 'none']

print(classification_report(y, y_pred, target_names=labels))

cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

