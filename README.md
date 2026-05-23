# Hệ thống nhận diện tín hiệu tay điều phối máy bay (Aircraft Marshaller Hand Signal Recognition)

Dự án AI nhận diện các tín hiệu tay điều phối máy bay (Aircraft Marshalling Hand Signals) sử dụng thư viện **MediaPipe Pose** và các mô hình Học máy/Học sâu (DNN & Random Forest). 

Hệ thống hỗ trợ 2 dạng ứng dụng chính:
1. **Ứng dụng GUI trên máy tính để bàn (PyQt6):** Cho phép bật camera trực tiếp (real-time) hoặc tải lên ảnh tĩnh để hiển thị khung xương và nhận diện tín hiệu ngay lập tức trên máy tính cá nhân.
2. **REST API & WebSocket Server (FastAPI + Socket.IO):** Nhận luồng dữ liệu hình ảnh (base64) từ các ứng dụng khách (như Next.js Frontend), nhận diện cử chỉ, đồng thời tính toán giả lập động học bay (tọa độ vị trí, vận tốc chuyển động) và phát telemetry ngược lại cho client.

---

## Các cử chỉ hỗ trợ (Labels)
1. **AHEAD** (Đi thẳng - Máy bay di chuyển về phía trước)
2. **RIGHT** (Rẽ phải - Máy bay rẽ phải theo góc nhìn của phi công)
3. **LEFT** (Rẽ trái - Máy bay rẽ trái theo góc nhìn của phi công)
4. **STOP** (Dừng lại - Dừng máy bay khẩn cấp/an toàn)
5. **NONE** (Không phát hiện tín hiệu)

*Lưu ý: Trong phần xử lý logic camera thời gian thực, cử chỉ `LEFT`/`RIGHT` được ánh xạ đảo ngược phù hợp với góc nhìn thực tế của người điều phối (Marshaller).*

---

## Cấu trúc dự án thực tế

```text
├── Dataset/                   # Thư mục chứa các file dữ liệu CSV gốc dùng cho huấn luyện
├── data_marshaller_final/     # Thư mục chứa các ảnh cử chỉ thô phân chia theo thư mục con nhãn
├── processed_npy_data/        # Bộ dữ liệu dạng numpy (.npy) sau khi chuẩn hóa (X_train, X_test, ...)
├── models/                    # Lưu trữ các mô hình đã huấn luyện (.keras, .pkl, classes.npy)
├── results/                   # Báo cáo kết quả huấn luyện (Ma trận nhầm lẫn, đồ thị accuracy/loss)
│   ├── dnn/
│   └── rf/
├── process_data.py            # Trích xuất landmarks từ ảnh thô ra file CSV lưu ở output_csv/
├── prepare_data.py            # Đọc CSV từ Dataset/, chuẩn hóa tương đối 24 tọa độ thân trên và lưu .npy
├── train_rf.py                # Huấn luyện mô hình Random Forest
├── train_dnn.py               # Huấn luyện mô hình mạng nơ-ron sâu (DNN) sử dụng Keras
├── app.py                     # Ứng dụng Desktop GUI giám sát độc lập (PyQt6)
├── main.py                    # API & Socket.IO Server chính của dự án (FastAPI)
├── requirements.txt           # Danh sách thư viện cần cài đặt
└── .env                       # Lưu cấu hình khóa bí mật của API
```

---

## Yêu cầu hệ thống & Cài đặt

### Yêu cầu
- Python 3.8+
- Webcam kết nối với máy tính (để chạy real-time)

### Các bước cài đặt

1. **Clone repository về máy:**
   ```bash
   git clone <URL_CUA_REPO>
   cd ramphash
   ```

2. **Tạo và kích hoạt môi trường ảo (Virtual Environment):**
   ```bash
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # Windows (CMD)
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Cấu hình biến môi trường:**
   - Tạo file `.env` ở thư mục gốc của dự án.
   - Thêm biến cấu hình bí mật dùng để xác thực API Key:
     ```env
     AI_SECRET_KEY=your_secret_key_here
     ```

---

## Quy trình chuẩn bị dữ liệu & Huấn luyện (Pipeline)

Nếu bạn muốn huấn luyện lại mô hình từ đầu, hãy thực hiện theo thứ tự sau:

1. **Trích xuất đặc trưng từ ảnh thô:**
   Đặt ảnh vào các thư mục con tương ứng bên trong `data_marshaller_final/` và chạy:
   ```bash
   python process_data.py
   ```
   Các file CSV lưu tọa độ xương thô của các khớp sẽ được sinh ra trong thư mục `output_csv/`.

2. **Tiền xử lý và chuẩn hóa dữ liệu:**
   Đảm bảo các file CSV huấn luyện nằm trong thư mục `Dataset/` (được copy từ `output_csv/` sang). Chạy lệnh:
   ```bash
   python prepare_data.py
   ```
   Script sẽ lấy 24 đặc trưng thân trên quan trọng (từ đầu đến cổ), chuẩn hóa tương đối tọa độ bằng cách lấy khớp Cổ làm gốc tọa độ `(0, 0)` và chia tỷ lệ theo khoảng cách vai để triệt tiêu ảnh hưởng của khoảng cách đứng xa/gần camera. Dữ liệu được chia tập train/test (80/20) và lưu vào `processed_npy_data/`.

3. **Huấn luyện mô hình:**
   - Huấn luyện **Random Forest**:
     ```bash
     python train_rf.py
     ```
     Mô hình sẽ lưu tại `models/random_forest_model.pkl`. Đồ thị độ quan trọng tính năng và ma trận nhầm lẫn sẽ được xuất ra `results/rf/`.
     
   - Huấn luyện **Deep Neural Network**:
     ```bash
     python train_dnn.py
     ```
     Mô hình mạng nơ-ron lưu tại `models/dnn_pose_model.keras`. Đồ thị độ chính xác (Accuracy), độ mất mát (Loss) và ma trận nhầm lẫn sẽ xuất ra `results/dnn/`.

---

## Hướng dẫn chạy các ứng dụng

### 1. Trải nghiệm giao diện Desktop GUI (PyQt6)
Chạy ứng dụng GUI giám sát độc lập, thân thiện:
```bash
python app.py
```
**Chức năng chính:**
- Chọn mô hình dự đoán mong muốn thông qua menu thả xuống: Random Forest hoặc DNN.
- Tab **Camera Realtime**: Bật/tắt camera trực tiếp từ Webcam, phân tích cử chỉ theo khung thời gian thực.
- Tab **Ảnh Tĩnh**: Tải lên một ảnh bất kỳ từ máy tính để phân tích cấu trúc xương pose landmarks và xuất kết quả nhận dạng kèm phần trăm độ tin cậy.
- Khung **Nhật ký giám sát hệ thống**: Hiển thị chi tiết xác suất đầu ra (raw probabilities) của từng nhãn lớp cử chỉ.

---

### 2. Khởi chạy API & Socket.IO Server (Phục vụ Web Client/Frontend)
Server chính chạy trên nền tảng **FastAPI**, tích hợp cổng giao tiếp Socket.IO hiệu năng cao hỗ trợ xử lý đa luồng qua ThreadPoolExecutor:
```bash
python main.py
```
Mặc định server sẽ chạy tại `http://localhost:8000`.

#### Tài liệu HTTP API (REST)
- **Endpoint:** `POST /predict-image`
- **Headers:** 
  - `x-api-key`: Phải trùng với khóa bí mật `AI_SECRET_KEY` trong file `.env`.
- **Body (JSON):**
  ```json
  {
    "image": "data:image/jpeg;base64,...",
    "model": "dnn",
    "frameId": 123
  }
  ```
- **Response (JSON):** Trả về cử chỉ nhận diện, độ tin cậy, tọa độ skeleton chuẩn hóa và frameId.

#### Tài liệu WebSocket Events (Socket.IO)
Kết nối trực tiếp tới namespace gốc `/` tại `ws://localhost:8000`.

- **Sự kiện gửi lên server (Listen Events):**
  - `start_session`: Bắt đầu phiên giả lập động học máy bay (nhận tham số `scenarioId`).
  - `pause_session` / `reset_session`: Tạm dừng hoặc đặt lại trạng thái máy bay về vị trí xuất phát.
  - `update_settings`: Cập nhật độ nhạy nhận diện AI (tham số `sensitivity` từ 0-100%).
  - `video_frame`: Gửi frame base64 từ camera của trình duyệt. 
    ```json
    {
      "frame": "data:image/jpeg;base64,...",
      "model": "dnn",
      "frameId": 456
    }
    ```
    *Mẹo: Server tự động áp dụng cửa sổ trượt trung bình (sliding average) 5 frames liên tục để làm mượt kết quả nhận diện cử chỉ, giảm nhiễu giật lag.*
  - `predict_image_static`: Gửi ảnh tĩnh tải lên từ web để phân tích xương.

- **Sự kiện nhận về từ server (Emit Events):**
  - `ai_result`: Kết quả nhận diện cử chỉ, tọa độ skeleton đầy đủ, và xác suất.
  - `predict_image_static_response`: Kết quả trả về cho ảnh tĩnh.
  - `telemetry_update`: Dữ liệu telemetry cập nhật động học của máy bay sau khi di chuyển dựa trên cử chỉ nhận diện.
    ```json
    {
      "x": 50.0,            // Tọa độ X hiện tại của máy bay (25.0 - 75.0)
      "y": 27.5,            // Tọa độ Y hiện tại của máy bay (tiến liên tục)
      "vx": -0.7,           // Vận tốc ngang (tác động bởi LEFT/RIGHT)
      "vy": 1.2,            // Vận tốc tiến (tác động bởi AHEAD, STOP, NONE)
      "gesture": "LEFT",    // Cử chỉ nhận diện được áp dụng làm mượt
      "confidence": 0.89,   // Độ tin cậy cử chỉ
      "accuracy": 0.87,     // Độ chính xác ước lượng
      "speed": 18,          // Vận tốc hiển thị quy đổi
      "elapsedTime": 42,    // Thời gian trôi qua của phiên giả lập (giây)
      "points": { ... },    // Các điểm khớp thân trên chính (head, neck, shoulders, elbows, wrists)
      "allPoints": [ ... ]  // Toàn bộ 33 điểm khớp xương MediaPipe
    }
    ```

---

## Ghi chú bảo mật & Git
Các tệp tin cấu hình môi trường `.env`, thư mục môi trường ảo `venv/`, các tập tin dữ liệu nhị phân nặng `.npy` và các file mô hình huấn luyện (`.keras`, `.pkl`) đã được liệt kê trong `.gitignore` để giữ kho lưu trữ gọn nhẹ và an toàn.
