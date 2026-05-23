import os
import sys
import cv2
import joblib
import numpy as np
import tensorflow as tf
import mediapipe as mp

from tensorflow.keras.models import load_model
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QComboBox, QFileDialog, QVBoxLayout, 
                             QHBoxLayout, QFrame, QTabWidget, QTextEdit)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

MODEL_DIR = "models"
DATA_DIR = "processed_npy_data"

COLUMNS_24 = [
    'noseX', 'noseY', 'left_eyeX', 'left_eyeY', 'right_eyeX', 'right_eyeY',
    'left_earX', 'left_earY', 'right_earX', 'right_earY',
    'left_shoulderX', 'left_shoulderY', 'right_shoulderX', 'right_shoulderY',
    'left_elbowX', 'left_elbowY', 'right_elbowX', 'right_elbowY',
    'left_wristX', 'left_wristY', 'right_wristX', 'right_wristY',
    'neckX', 'neckY'
]

DARK_STYLE = """
    QMainWindow { background-color: #121212; }
    QWidget { color: #E0E0E0; font-family: 'Segoe UI', Arial, sans-serif; }
    QFrame#Sidebar { background-color: #1E1E1E; border-right: 1px solid #2D2D2D; }
    QTabWidget::pane { border: 1px solid #2D2D2D; background-color: #121212; }
    QTabBar::tab { background: #1E1E1E; color: #888888; padding: 10px 20px; border: 1px solid #2D2D2D; border-bottom: none; }
    QTabBar::tab:selected { background: #2D2D2D; color: #00E676; font-weight: bold; }
    QPushButton { background-color: #2979FF; color: white; border: none; padding: 10px 15px; border-radius: 5px; font-weight: bold; }
    QPushButton:hover { background-color: #2962FF; }
    QPushButton#StopBtn { background-color: #FF1744; }
    QPushButton#StopBtn:hover { background-color: #D50000; }
    QPushButton#RepredictBtn { background-color: #00E676; color: #121212; }
    QPushButton#RepredictBtn:hover { background-color: #00C853; }
    QComboBox { background-color: #2D2D2D; border: 1px solid #444444; padding: 8px; border-radius: 5px; color: white; }
    QLabel#ImgScreen { background-color: #000000; border: 2px dashed #333333; border-radius: 8px; }
    QTextEdit { background-color: #151515; border: 1px solid #2D2D2D; color: #00FF66; font-family: 'Consolas', Monaco, monospace; font-size: 11px; }
"""

def normalize_upper_body_features(features_2d):
    norm_features = features_2d.copy()
    for i in range(len(norm_features)):
        row = norm_features[i]
        neck_x, neck_y = row[22], row[23]
        if neck_x == 0 and neck_y == 0:
            continue
        ls_x, ls_y = row[10], row[11]
        rs_x, rs_y = row[12], row[13]
        shoulder_dist = np.sqrt((ls_x - rs_x)**2 + (ls_y - rs_y)**2)
        if shoulder_dist == 0:
            shoulder_dist = 1.0
        for j in range(0, 24):
            if j % 2 == 0:
                row[j] = (row[j] - neck_x) / shoulder_dist
            else:
                row[j] = (row[j] - neck_y) / shoulder_dist
        norm_features[i] = row
    return norm_features

# Luồng chạy nền để chụp frame camera và chạy dự báo mà không khóa luồng UI
class CameraWorker(QThread):
    # Tín hiệu gửi frame và kết quả phân loại về luồng chính
    image_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, float, list, list)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.running = False
        self.latest_frame = None  

    def run(self):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        # Khởi tạo mô hình Pose trong chế độ luồng video (static_image_mode=False nhanh hơn do theo dõi điểm khớp)
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0)

        # Mở webcam để đọc luồng hình ảnh
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        frame_count = 0
        
        last_label = "WAITING..."
        last_conf = 0.0
        last_probs = []
        last_features = []
        last_landmarks = None

        while self.running:
            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            
            self.latest_frame = frame.copy()  
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            
            if frame_count % 2 == 0:
                results = pose.process(image_rgb)
                last_landmarks = results.pose_landmarks

                if results.pose_landmarks:
                    row_data = {col: 0.0 for col in COLUMNS_24}
                    landmarks = results.pose_landmarks.landmark
                    
                    row_data['noseX'], row_data['noseY'] = landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y
                    row_data['left_eyeX'], row_data['left_eyeY'] = landmarks[mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE].y
                    row_data['right_eyeX'], row_data['right_eyeY'] = landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y
                    row_data['left_earX'], row_data['left_earY'] = landmarks[mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR].y
                    row_data['right_earX'], row_data['right_earY'] = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y
                    row_data['left_shoulderX'], row_data['left_shoulderY'] = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                    row_data['right_shoulderX'], row_data['right_shoulderY'] = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                    row_data['left_elbowX'], row_data['left_elbowY'] = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
                    row_data['right_elbowX'], row_data['right_elbowY'] = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
                    row_data['left_wristX'], row_data['left_wristY'] = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
                    row_data['right_wristX'], row_data['right_wristY'] = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
                    row_data['neckX'] = (row_data['left_shoulderX'] + row_data['right_shoulderX']) / 2
                    row_data['neckY'] = (row_data['left_shoulderY'] + row_data['right_shoulderY']) / 2
                    
                    features_2d = np.array([list(row_data.values())])
                    features_2d_scaled = normalize_upper_body_features(features_2d)
                    last_features = features_2d_scaled[0].tolist()

                    # Chạy dự đoán sử dụng mô hình được lựa chọn động trong hộp thoại
                    if self.main_window.model is not None:
                        if self.main_window.current_model_type == "RF":
                            raw_probs = self.main_window.model.predict_proba(features_2d_scaled)[0]
                        else:
                            raw_probs = self.main_window.model.predict(features_2d_scaled, verbose=0)[0]
                        
                        pred_idx = np.argmax(raw_probs)
                        last_conf = raw_probs[pred_idx] * 100
                        last_probs = raw_probs.tolist()
                        last_label = self.main_window.classes[pred_idx]
                else:
                    last_label = "WAITING..."
                    last_conf = 0.0
                    last_probs = []
                    last_features = []

            image_disp_rgb = image_rgb.copy()
            
            # Vẽ đè các đường khung xương trên frame
            if last_landmarks:
                mp_drawing.draw_landmarks(
                    image_disp_rgb, last_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

            # Phát tín hiệu dữ liệu về luồng GUI chính
            self.image_signal.emit(image_disp_rgb)
            self.result_signal.emit(last_label, last_conf, last_probs, last_features)
            self.msleep(15)

        cap.release()
        pose.close()

    def stop(self):
        self.running = False
        self.wait()

# Cửa sổ ứng dụng chính
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marshalling Pose System - Premium Sync Monitor")
        self.setGeometry(100, 100, 1250, 750)
        self.setStyleSheet(DARK_STYLE)

        self.model = None
        self.current_model_type = "RF"
        self.latest_static_image = None  

        # Nạp danh sách nhãn lớp
        if os.path.exists(os.path.join(DATA_DIR, 'classes.npy')):
            self.classes = np.load(os.path.join(DATA_DIR, 'classes.npy'), allow_pickle=True)
        else:
            self.classes = ["AHEAD", "LEFT", "RIGHT", "STOP", "NONE"]

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0)

        # Khởi tạo luồng camera worker
        self.camera_thread = CameraWorker(self)
        self.camera_thread.image_signal.connect(self.update_live_video)
        self.camera_thread.result_signal.connect(self.update_live_result)

        self.initUI()
        self.change_model()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Xây dựng bảng điều khiển Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 25, 20, 25)
        sidebar_layout.setSpacing(12)

        lbl_logo = QLabel("<h2>MARSHALLING AI</h2>")
        lbl_logo.setStyleSheet("color: #00E676; font-weight: bold; letter-spacing: 1px;")
        
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Random Forest (.pkl)", "Deep Neural Network (.keras)"])
        self.combo_model.currentIndexChanged.connect(self.change_model)

        self.btn_repredict = QPushButton("Nhận diện lại (Re-predict)")
        self.btn_repredict.setObjectName("RepredictBtn")
        self.btn_repredict.clicked.connect(self.trigger_repredict)

        result_box = QFrame()
        result_box.setStyleSheet("background-color: #252525; border-radius: 8px; padding: 12px;")
        result_box_layout = QVBoxLayout(result_box)
        
        lbl_res_title = QLabel("TƯ THẾ NHẬN DIỆN:")
        lbl_res_title.setStyleSheet("font-size: 11px; color: #888888; font-weight: bold;")
        
        self.lbl_class_output = QLabel("NONE")
        self.lbl_class_output.setStyleSheet("font-size: 28px; color: #00E676; font-weight: bold;")
        
        self.lbl_conf_output = QLabel("Độ tin cậy: 0.00%")
        self.lbl_conf_output.setStyleSheet("font-size: 13px; color: #B0BEC5;")

        result_box_layout.addWidget(lbl_res_title)
        result_box_layout.addWidget(self.lbl_class_output)
        result_box_layout.addWidget(self.lbl_conf_output)

        sidebar_layout.addWidget(lbl_logo)
        sidebar_layout.addWidget(QLabel("<b>Bộ phân loại:</b>"))
        sidebar_layout.addWidget(self.combo_model)
        sidebar_layout.addWidget(self.btn_repredict)  
        sidebar_layout.addWidget(result_box)
        
        sidebar_layout.addWidget(QLabel("<b>📊 Thông Số Đầu Ra Gốc:</b>"))
        self.txt_debug_log = QTextEdit()
        self.txt_debug_log.setReadOnly(True)
        sidebar_layout.addWidget(self.txt_debug_log, stretch=1)

        main_layout.addWidget(sidebar, stretch=1)

        self.tabs = QTabWidget()
        
        tab_realtime = QWidget()
        rt_layout = QVBoxLayout(tab_realtime)
        self.lbl_video_screen = QLabel("Bấm nút 'Bật Camera Realtime' để khởi chạy luồng video")
        self.lbl_video_screen.setObjectName("ImgScreen")
        self.lbl_video_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video_screen.setFixedSize(640, 480)
        
        btn_box = QHBoxLayout()
        self.btn_start_cam = QPushButton("Bật Camera")
        self.btn_start_cam.clicked.connect(self.start_camera)
        self.btn_stop_cam = QPushButton("Tắt Camera")
        self.btn_stop_cam.setObjectName("StopBtn")
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_stop_cam.setEnabled(False)
        btn_box.addWidget(self.btn_start_cam)
        btn_box.addWidget(self.btn_stop_cam)
        
        rt_layout.addWidget(self.lbl_video_screen, alignment=Qt.AlignmentFlag.AlignCenter)
        rt_layout.addLayout(btn_box)

        tab_static = QWidget()
        static_layout = QVBoxLayout(tab_static)
        img_viewers_layout = QHBoxLayout()
        
        self.lbl_img_original = QLabel("Ảnh gốc tải lên")
        self.lbl_img_original.setObjectName("ImgScreen")
        self.lbl_img_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_original.setFixedSize(400, 400)
        
        self.lbl_img_pose = QLabel("Ảnh phân tích khung xương")
        self.lbl_img_pose.setObjectName("ImgScreen")
        self.lbl_img_pose.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_pose.setFixedSize(400, 400)
        
        img_viewers_layout.addWidget(self.lbl_img_original)
        img_viewers_layout.addWidget(self.lbl_img_pose)
        
        self.btn_upload = QPushButton("📸 Chọn ảnh tĩnh...")
        self.btn_upload.clicked.connect(self.upload_image)
        
        static_layout.addLayout(img_viewers_layout)
        static_layout.addWidget(self.btn_upload)

        self.tabs.addTab(tab_realtime, "🎬 Camera Realtime")
        self.tabs.addTab(tab_static, "🖼️ Ảnh Tĩnh")

        main_layout.addWidget(self.tabs, stretch=2)

    def change_model(self):
        model_name = self.combo_model.currentText()
        try:
            if model_name == "Random Forest (.pkl)":
                path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
                self.model = joblib.load(path)
                self.current_model_type = "RF"
            else:
                path = os.path.join(MODEL_DIR, 'dnn_pose_model.keras')
                self.model = load_model(path)
                self.current_model_type = "DNN"
            self.lbl_class_output.setText("READY")
            self.lbl_conf_output.setText(f"Đã nạp: {model_name.split()[0]}")
            self.trigger_repredict()
        except Exception as e:
            self.lbl_class_output.setText("ERR")
            self.lbl_conf_output.setText("Không tìm thấy tệp model")

    def trigger_repredict(self):
        current_tab_idx = self.tabs.currentIndex()
        if current_tab_idx == 0:
            if self.camera_thread.running and self.camera_thread.latest_frame is not None:
                self.execute_direct_predict(self.camera_thread.latest_frame)
        elif current_tab_idx == 1:
            if self.latest_static_image is not None:
                # Ảnh tĩnh truyền thẳng vào không lật
                self.execute_direct_predict(self.latest_static_image)

    def execute_direct_predict(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        row_data = {col: 0.0 for col in COLUMNS_24}
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row_data['noseX'], row_data['noseY'] = landmarks[self.mp_pose.PoseLandmark.NOSE].x, landmarks[self.mp_pose.PoseLandmark.NOSE].y
            row_data['left_eyeX'], row_data['left_eyeY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EYE].y
            row_data['right_eyeX'], row_data['right_eyeY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE].y
            row_data['left_earX'], row_data['left_earY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y
            row_data['right_earX'], row_data['right_earY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].y
            row_data['left_shoulderX'], row_data['left_shoulderY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            row_data['right_shoulderX'], row_data['right_shoulderY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            row_data['left_elbowX'], row_data['left_elbowY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y
            row_data['right_elbowX'], row_data['right_elbowY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
            row_data['left_wristX'], row_data['left_wristY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
            row_data['right_wristX'], row_data['right_wristY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
            row_data['neckX'] = (row_data['left_shoulderX'] + row_data['right_shoulderX']) / 2
            row_data['neckY'] = (row_data['left_shoulderY'] + row_data['right_shoulderY']) / 2
            
            features_2d = np.array([list(row_data.values())])
            features_2d_scaled = normalize_upper_body_features(features_2d)
            
            if self.model is not None:
                if self.current_model_type == "RF":
                    raw_probs = self.model.predict_proba(features_2d_scaled)[0]
                else:
                    raw_probs = self.model.predict(features_2d_scaled, verbose=0)[0]
                
                pred_idx = np.argmax(raw_probs)
                label_result = self.classes[pred_idx]
                conf = raw_probs[pred_idx] * 100
                
                self.lbl_class_output.setText(label_result.upper())
                self.lbl_conf_output.setText(f"Độ tin cậy: {conf:.2f}%")
                self.update_live_result(label_result, conf, raw_probs.tolist(), features_2d_scaled[0].tolist())

    def start_camera(self):
        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)
        self.camera_thread.start()

    def stop_camera(self):
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.camera_thread.stop()
        self.lbl_video_screen.clear()
        self.lbl_video_screen.setText("Đã tắt camera.")

    def update_live_video(self, img_array):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.lbl_video_screen.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.lbl_video_screen.setPixmap(scaled_pixmap)

    def update_live_result(self, label_result, confidence, probabilities, scaled_features):
        if label_result == "WAITING...":
            self.lbl_class_output.setText("SCANNING")
            self.lbl_conf_output.setText("Đang tìm dáng người...")
            self.txt_debug_log.setText("MediaPipe Pose: Khuyết người...")
        else:
            self.lbl_class_output.setText(label_result.upper())
            self.lbl_conf_output.setText(f"Độ tin cậy: {confidence:.2f}%")
            
            debug_str = f"=== SYSTEM MONITOR LOG ===\n"
            debug_str += f"Classifier : {self.combo_model.currentText().split()[0]}\n"
            debug_str += f"Decision   : {label_result.upper()} ({confidence:.2f}%)\n"
            debug_str += f"--- RAW PROBABILITIES ---\n"
            for idx, class_name in enumerate(self.classes):
                prob_val = probabilities[idx] if idx < len(probabilities) else 0.0
                debug_str += f"[{idx}] {class_name.upper():<8}: {prob_val*100:.2f}%\n"
            self.txt_debug_log.setText(debug_str)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn hình ảnh tư thế tĩnh", "", "Image Files (*.png *.jpg *.jpeg *.webp)")
        if not file_path: return
            
        image = cv2.imread(file_path)
        if image is None: return
        
        self.latest_static_image = image.copy()  
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.display_static_image(image_rgb, self.lbl_img_original)
        
        results = self.pose.process(image_rgb)
        row_data = {col: 0.0 for col in COLUMNS_24}
        
        if results.pose_landmarks:
            annotated_image = image_rgb.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            self.display_static_image(annotated_image, self.lbl_img_pose)
            
            landmarks = results.pose_landmarks.landmark
            row_data['noseX'], row_data['noseY'] = landmarks[self.mp_pose.PoseLandmark.NOSE].x, landmarks[self.mp_pose.PoseLandmark.NOSE].y
            row_data['left_eyeX'], row_data['left_eyeY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EYE].y
            row_data['right_eyeX'], row_data['right_eyeY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE].y
            row_data['left_earX'], row_data['left_earY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y
            row_data['right_earX'], row_data['right_earY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].y
            row_data['left_shoulderX'], row_data['left_shoulderY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            row_data['right_shoulderX'], row_data['right_shoulderY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            row_data['left_elbowX'], row_data['left_elbowY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y
            row_data['right_elbowX'], row_data['right_elbowY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
            row_data['left_wristX'], row_data['left_wristY'] = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
            row_data['right_wristX'], row_data['right_wristY'] = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
            row_data['neckX'] = (row_data['left_shoulderX'] + row_data['right_shoulderX']) / 2
            row_data['neckY'] = (row_data['left_shoulderY'] + row_data['right_shoulderY']) / 2
            
            features_2d = np.array([list(row_data.values())])
            features_2d_scaled = normalize_upper_body_features(features_2d)
            
            if self.model is not None:
                if self.current_model_type == "RF":
                    raw_probs = self.model.predict_proba(features_2d_scaled)[0]
                else:
                    raw_probs = self.model.predict(features_2d_scaled, verbose=0)[0]
                    
                pred_idx = np.argmax(raw_probs)
                label_result = self.classes[pred_idx]
                conf = raw_probs[pred_idx] * 100
                self.lbl_class_output.setText(label_result.upper())
                self.lbl_conf_output.setText(f"Độ tin cậy: {conf:.2f}%")
                
                debug_str = f"=== STATIC IMAGE RESULTS ===\n"
                for idx, class_name in enumerate(self.classes):
                    debug_str += f"[{idx}] {class_name.upper():<8}: {raw_probs[idx]*100:.2f}%\n"
                self.txt_debug_log.setText(debug_str)
        else:
            self.lbl_img_pose.setText("Không tìm thấy dáng người")

    def display_static_image(self, img_array, label_widget):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(label_widget.width(), label_widget.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())