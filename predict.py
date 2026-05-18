import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# --- CẤU HÌNH GIAO DIỆN STREAMLIT ---
st.set_page_config(
    page_title="✈️ Marshaller Signal AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

LABELS = ['AHEAD', 'RIGHT', 'LEFT', 'STOP', 'NONE']

# --- HÀM TẢI MÔ HÌNH (Sử dụng Cache để tránh load lại làm chậm web) ---
@st.cache_resource
def load_all_models():
    dnn = load_model('models/marshaller_model_dnn.h5')
    rf = joblib.load('models/marshaller_model_rf.pkl')
    scaler = joblib.load('models/marshaller_scaler.pkl') if os.path.exists('models/marshaller_scaler.pkl') else None
    return dnn, rf, scaler

# Nạp các mô hình vào bộ nhớ
dnn_model, rf_model, scaler_model = load_all_models()

# Khởi tạo giải pháp nhận diện dáng người MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- GIAO DIỆN CHÍNH ---
st.title("✈️ Marshaller Signal Recognition System")
st.markdown("Hệ thống nhận diện hiệu lệnh điều hướng máy bay/xe cộ thời gian thực bằng Trí tuệ nhân tạo.")

# Thanh cấu hình bên trái (Sidebar)
st.sidebar.header("⚙️ Cấu hình hệ thống")
model_choice = st.sidebar.radio(
    "1. Chọn thuật toán mô hình:", 
    ("DNN (Deep Learning)", "Random Forest (Machine Learning)")
)
input_choice = st.sidebar.selectbox(
    "2. Chọn nguồn đầu vào dữ liệu:", 
    ("Tải ảnh lên", "Webcam trực tiếp")
)

# --- HÀM CORE XỬ LÝ AI VÀ DỰ ĐOÁN ---
def predict(frame, model, mode):
    # Chuyển ảnh từ BGR (OpenCV) sang RGB (MediaPipe)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    
    if res.pose_landmarks:
        # Trích xuất 132 đặc trưng (33 điểm mốc x 4 giá trị)
        lms = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
        features = np.array(lms).flatten().reshape(1, -1)
        
        if mode == "dnn":
            # BẮT BUỘC: Chuẩn hóa dữ liệu bằng scaler nếu chọn chạy mô hình DNN
            if scaler_model:
                features = scaler_model.transform(features)
            pred = model.predict(features, verbose=0)
            idx = np.argmax(pred)
            conf = float(pred[0][idx])
        else:
            # Mô hình Random Forest không cần scale dữ liệu
            idx = int(model.predict(features)[0])
            conf = float(np.max(model.predict_proba(features)))
            
        # Vẽ các kết nối xương lên ma trận ảnh
        mp.solutions.drawing_utils.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        return LABELS[idx], conf, frame
        
    return "NONE", 0.0, frame

# --- PHÂN LUỒNG XỬ LÝ THEO GIAO DIỆN ---

# Trường hợp 1: Người dùng chọn tải ảnh tĩnh từ máy lên
if input_choice == "Tải ảnh lên":
    st.subheader("📸 Phân tích dữ liệu qua hình ảnh tĩnh")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn file ảnh từ máy của bạn...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Đọc dữ liệu ảnh byte sang mảng OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Lựa chọn model theo cấu hình sidebar
        selected_model = dnn_model if "DNN" in model_choice else rf_model
        mode = "dnn" if "DNN" in model_choice else "rf"
        
        # Gọi hàm AI xử lý
        label, conf, processed_img = predict(image, selected_model, mode)
        
        # Hiển thị song song kết quả trên giao diện web
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Hình ảnh đã vẽ tọa độ mốc xương")
        with col2:
            st.info("📊 **KẾT QUẢ PHÂN TÍCH CHI TIẾT**")
            st.subheader(f"Hành động nhận diện: `{label}`")
            st.metric(label="Độ tin cậy của thuật toán", value=f"{conf * 100:.2f}%")

# Trường hợp 2: Người dùng chọn mở Camera thời gian thực
else:
    st.subheader("🎥 Nhận diện thời gian thực qua Webcam")
    st.warning("Ứng dụng trên trình duyệt yêu cầu bạn cấp quyền sử dụng thiết bị Camera.")
    
    run = st.checkbox('Kích hoạt mở Camera')
    FRAME_WINDOW = st.image([]) # Tạo khung trống để liên tục cập nhật frame video
    
    if run:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret: 
                st.error("Không thể kết nối hoặc đọc luồng video từ Webcam.")
                break
            
            # Lật ngược ảnh theo chiều ngang để người dùng nhìn giống như đang soi gương
            frame = cv2.flip(frame, 1)
            
            # Kiểm tra cấu hình thuật toán được chọn
            selected_model = dnn_model if "DNN" in model_choice else rf_model
            mode = "dnn" if "DNN" in model_choice else "rf"
            
            # Xử lý AI qua từng khung hình (Frame)
            label, conf, processed_img = predict(frame, selected_model, mode)
            
            # Quy định màu chữ (Nếu hiệu lệnh là STOP thì đổi sang màu đỏ để cảnh báo)
            color_text = (0, 0, 255) if label == 'STOP' else (0, 255, 0)
            
            # Ghi trực tiếp nhãn kết quả lên khung hình video
            cv2.putText(
                processed_img, f"Hieu lenh: {label} ({conf*100:.1f}%)", (15, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv2.LINE_AA
            )
            
            # Cập nhật frame mới lên màn hình giao diện web Streamlit
            FRAME_WINDOW.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            
        # Giải phóng thiết bị camera ngay lập tức khi người dùng bỏ chọn tắt Camera
        cap.release()
    else:
        st.info("Thiết bị Camera hiện đang đóng. Hãy tích vào nút phía trên để bắt đầu stream.")