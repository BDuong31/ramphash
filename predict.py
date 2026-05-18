import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# --- CẤU HÌNH ---
st.set_page_config(page_title="Marshaller AI", layout="wide")
LABELS = ['AHEAD', 'RIGHT', 'LEFT', 'STOP', 'NONE']

@st.cache_resource
def load_all_models():
    dnn = load_model('models/marshaller_model_dnn.h5')
    rf = joblib.load('mnodels/marshaller_model_rf.pkl')
    return dnn, rf

dnn_model, rf_model = load_all_models()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7)

# --- GIAO DIỆN ---
st.title("✈️ Marshaller Signal Recognition System")
st.sidebar.header("Cấu hình")

model_choice = st.sidebar.radio("Chọn mô hình:", ("DNN (Deep Learning)", "Random Forest"))
input_choice = st.sidebar.selectbox("Nguồn đầu vào:", ("Tải ảnh lên", "Webcam trực tiếp"))

def predict(frame, model, mode):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if res.pose_landmarks:
        lms = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
        features = np.array(lms).flatten().reshape(1, -1)
        
        if mode == "dnn":
            pred = model.predict(features, verbose=0)
            idx = np.argmax(pred)
            conf = pred[0][idx]
        else:
            idx = model.predict(features)[0]
            conf = np.max(model.predict_proba(features))
            
        # Vẽ lên frame
        mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return LABELS[idx], conf, frame
    return None, 0, frame

# --- XỬ LÝ ĐẦU VÀO ---
if input_choice == "Tải ảnh lên":
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        selected_model = dnn_model if "DNN" in model_choice else rf_model
        mode = "dnn" if "DNN" in model_choice else "rf"
        
        label, conf, processed_img = predict(image, selected_model, mode)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Kết quả phân tích")
        with col2:
            st.subheader("Kết quả dự đoán:")
            st.write(f"**Hành động:** {label}")
            st.write(f"**Độ tin cậy:** {conf*100:.2f}%")

else:
    st.warning("Webcam trên trình duyệt yêu cầu quyền truy cập Camera.")
    run = st.checkbox('Mở Camera')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        selected_model = dnn_model if "DNN" in model_choice else rf_model
        mode = "dnn" if "DNN" in model_choice else "rf"
        
        label, conf, processed_img = predict(frame, selected_model, mode)
        
        # Hiển thị thông tin lên màn hình web
        cv2.putText(processed_img, f"{label} {conf*100:.1f}%", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        FRAME_WINDOW.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    else:
        st.info("Đã dừng Camera.")