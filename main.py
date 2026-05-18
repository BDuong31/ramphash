import cv2
import mediapipe as mp
import numpy as np
import joblib
import socketio
import uvicorn
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException, Depends
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

# --- CẤU HÌNH BẢO MẬT ---
AI_SECRET_KEY = os.getenv("AI_SECRET_KEY")

# --- LOAD MODELS ---
LABELS = ['AHEAD', 'RIGHT', 'LEFT', 'STOP', 'NONE']
dnn_model = load_model('models/marshaller_model_dnn.h5')
rf_model = joblib.load('models/marshaller_model_rf.pkl')

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7)

# Khởi tạo ThreadPool để xử lý các tác vụ nặng (CPU-bound) như AI và decode ảnh
executor = ThreadPoolExecutor(max_workers=4) 

# --- KHỞI TẠO SERVER ---
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = socketio.ASGIApp(sio)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOGIC XỬ LÝ AI (Hàm đồng bộ thuần túy) ---
def process_ai_sync(frame, model_type="dnn"):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    
    if res.pose_landmarks:
        landmarks_list = []
        for lm in res.pose_landmarks.landmark:
            landmarks_list.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility)
            })
        lms = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
        features = np.array(lms).flatten().reshape(1, -1)
        
        if model_type == "dnn":
            pred = dnn_model.predict(features, verbose=0)
            idx = np.argmax(pred)
            conf = float(pred[0][idx])
        else:
            idx = int(rf_model.predict(features)[0])
            conf = float(np.max(rf_model.predict_proba(features)))
            
        return {
            "label": LABELS[idx], 
            "confidence": conf,
            "landmarks": landmarks_list # Ngăn nắp từ 0 -> 32
        }
    return {"label": "NONE", "confidence": 0.0, "landmarks": []}


# --- HÀM BỔ TRỢ GIẢI MÃ ẢNH (Tránh block thread chính) ---
def decode_base64_image(base64_string):
    try:
        encoded_data = base64_string.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# --- MIDDLEWARE XÁC THỰC ---
def verify_secret_key(request: Request):
    key = request.headers.get("x-api-key")
    if key != AI_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid Key")

# --- ROUTES HTTP (Cho Axios) ---
@app.post("/predict-image")
async def predict_image(data: dict, _=Depends(verify_secret_key)):
    loop = asyncio.get_running_loop()
    
    # Chạy decode ảnh và predict trong ThreadPool để không nghẽn Event Loop
    frame = await loop.run_in_executor(executor, decode_base64_image, data.get("image"))
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
        
    result = await loop.run_in_executor(executor, process_ai_sync, frame, data.get("model", "dnn"))
    return result

# --- ROUTES SOCKET.IO (Cho Realtime) ---
@sio.event
async def connect(sid, environ, auth):
    token = auth.get("token") if auth else None
    
    if not token:
        import urllib.parse
        query_string = environ.get('QUERY_STRING', '')
        params = urllib.parse.parse_qs(query_string)
        token = params.get('token', [None])[0]

    if token == AI_SECRET_KEY:
        print(f"Authorized sid: {sid}")
        return True
    else:
        print(f"Connection refused for sid: {sid}")
        return False

@sio.on("stream_frame")
async def handle_stream(sid, data):
    """
    Nhận frame từ webcam dạng phi tập trung (Non-blocking) nhờ ThreadPoolExecutor
    """
    try:
        loop = asyncio.get_running_loop()
        
        # 1. Giải mã ảnh phi đồng bộ
        frame = await loop.run_in_executor(executor, decode_base64_image, data["image"])
        if frame is None:
            return

        # 2. Xử lý AI phi đồng bộ (Chạy ngầm trong ThreadPool)
        result = await loop.run_in_executor(executor, process_ai_sync, frame, data.get("model", "dnn"))
        
        # 3. Gửi kết quả về ngay lập tức cho đúng client
        await sio.emit("ai_result", result, to=sid)
    except Exception as e:
        print(f"Error in stream handling: {e}")

app.mount("/", socket_app)

if __name__ == "__main__":
    # Bật uvloop nếu chạy trên Linux để tối ưu hóa IO tốc độ cao
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)