# Aircraft Marshaller Hand Signal Recognition System

Dự án AI nhận diện các tín hiệu tay điều hướng máy bay (Aircraft Marshalling signals) sử dụng thư viện **MediaPipe Pose** và các mô hình Học máy/Học sâu (DNN & Random Forest).
Hệ thống cung cấp dịch vụ dự đoán qua giao thức HTTP (REST API) và nhận luồng dữ liệu thời gian thực (Real-time) qua Socket.IO được xây dựng trên **FastAPI**.

## Các tín hiệu hỗ trợ (Labels)
1. **AHEAD** (Đi thẳng)
2. **RIGHT** (Rẽ phải)
3. **LEFT** (Rẽ trái)
4. **STOP** (Dừng lại)
5. **NONE** (Không có tín hiệu)

## Cấu trúc dự án
- `main.py`: Khởi chạy server FastAPI & Socket.IO để phục vụ dự đoán.
- `create_dataset.py`: Script trích xuất đặc trưng (pose landmarks) từ hình ảnh để tạo dataset `.npy`.
- `train_model_dnn.py` / `train_model_rf.py`: Script huấn luyện mô hình phân loại với DNN (TensorFlow/Keras) hoặc Random Forest (Scikit-Learn).
- `test_model_*.py`: Các file kiểm tra và đánh giá mô hình.
- `predict_realtime_*.py`: Script mô phỏng việc dự đoán tín hiệu trực tiếp từ webcam.
- `visualize_data.py`: Hỗ trợ hiển thị trực quan hoá dữ liệu bộ xương của MediaPipe.

## Yêu cầu hệ thống
- Python 3.8+
- Webcam (nếu muốn sử dụng realtime)

## Cài đặt (Installation)

1. Clone repository về máy của bạn:
   ```bash
   git clone <URL_CUA_REPO>
   cd ramphash
   ```

2. Tạo và kích hoạt môi trường ảo (Virtual Environment):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

4. Cấu hình biến môi trường:
   - Tạo file `.env` ở thư mục gốc của dự án.
   - Thêm cấu hình biến môi trường sau:
     ```env
     AI_SECRET_KEY=your_secret_key_here
     ```

## Sử dụng (Usage)

### Khởi động API Server
```bash
python main.py
```
Server mặc định sẽ lắng nghe ở địa chỉ `http://0.0.0.0:8000`.

### Giao tiếp HTTP (REST)
- **Endpoint**: `POST /predict-image`
- **Headers**: Yêu cầu xác thực qua header `x-api-key`, giá trị tương ứng với biến `AI_SECRET_KEY` trong file `.env`.
- **Body JSON**: Gửi dữ liệu ảnh dạng `base64`.

### Giao tiếp WebSocket (Socket.IO)
- Kết nối tới `ws://localhost:8000` (Socket.IO namespace `/`).
- **Xác thực**: Truyền tham số token khi kết nối hoặc qua Query String.
- **Sự kiện Emit**: `stream_frame` với dữ liệu base64 ảnh.
- **Sự kiện Listen**: `ai_result` để nhận kết quả nhãn dự đoán và độ tự tin (confidence).

## Ghi chú
Các tệp dữ liệu dung lượng lớn (như `*.npy`), môi trường `venv/`, và các file mô hình (model) lớn đã được bỏ qua trong file `.gitignore` để tránh tràn dung lượng GitHub. Nếu cần chia sẻ các model `.h5`, bạn có thể nén chúng lại hoặc sử dụng Git LFS.
