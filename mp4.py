import ssl
import sys
import os
import torch
import cv2
import numpy as np
import json
import requests
import threading
from datetime import datetime
from flask import Flask, request, jsonify

# === Flask 서버 설정 ===
app = Flask(__name__)

# GPS 데이터를 전역 변수로 설정
gps_data = {
    "latitude": 37.123,  # 초기값
    "longitude": 127.123  # 초기값
}

@app.route('/gps', methods=['GET'])
def get_gps():
    # GET 요청에 GPS 데이터를 반환
    return jsonify(gps_data), 200

@app.route('/gps', methods=['POST'])
def set_gps():
    data = request.json
    location = data.get('location').split(',')
    gps_data["latitude"] = float(location[0])
    gps_data["longitude"] = float(location[1])
    speed = data.get('speed')

    print(f"Latitude: {gps_data['latitude']}, Longitude: {gps_data['longitude']}, Speed: {speed}")
    return "Received", 200

# Flask 서버를 백그라운드에서 실행하는 함수
def run_flask_server():
    app.run(host='0.0.0.0', port=5000)

# === YOLO 및 포트홀 등록 코드 ===

# === 환경설정 ===
API_BASE_URL = "http://43.202.82.198:8080"  # API 서버의 기본 URL (나중에 바꾸기 쉽게 전역변수로 설정)
POTHOLES_URI = f"{API_BASE_URL}/potholes"  # 포트홀 등록 URI
PRESIGNED_URL_URI = f"{API_BASE_URL}/presigned-url"  # Presigned URL 요청 URI

# YOLOv5 디렉토리를 PYTHONPATH에 추가
yolov5_path = os.path.join(os.getcwd(), 'yolov5')
sys.path.append(yolov5_path)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# YOLOv5 모델 로드 (로컬 파일 사용)
model_path = os.path.join(os.getcwd(), 'best.pt')
device = select_device('')
model = attempt_load(model_path)
model = model.cuda()
model.eval()

# 클래스 이름 로드
class_names = model.module.names if hasattr(model, 'module') else model.names

# 비디오 파일 설정
video_path = os.path.join(os.getcwd(), 'bundang.mp4')
cap = cv2.VideoCapture(video_path)

# S3에 이미지 업로드 함수
def upload_image_to_s3(image_path, presigned_url):
    with open(image_path, 'rb') as image_file:
        response = requests.put(presigned_url, data=image_file)
        if response.status_code == 200:
            print(f"Successfully uploaded {image_path}")
        else:
            print(f"Failed to upload {image_path}")

# 포트홀 정보 등록 함수
def register_pothole_info(file_url, latitude, longitude):
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "imageUrl": file_url  # S3에 저장된 이미지 URL
    }

    headers = {'Content-Type': 'application/json'}

    # API 서버에 POST 요청 보내기
    response = requests.post(POTHOLES_URI, json=payload, headers=headers)
    if response.status_code == 200:
        print(f"Pothole information successfully registered with image: {file_url}")
    else:
        print(f"Failed to register pothole information, status code: {response.status_code}")

# presigned URL 가져오는 함수
def get_presigned_url():
    response = requests.get(PRESIGNED_URL_URI)
    if response.status_code == 200:
        data = response.json()
        return data['presignedUrl'], data['fileUrl']
    else:
        print("Failed to get presigned URL")
        return None, None

# 이미지 및 검출 정보 저장 함수
def save_detection_info(frame, detections, save_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    image_path = f"{save_path}/{timestamp}.jpg"
    json_path = f"{save_path}/{timestamp}.json"

    # 이미지 저장
    image_saved = cv2.imwrite(image_path, frame)
    if not image_saved:
        print(f"Failed to save image: {image_path}")
        return None

    # JSON 데이터 생성 및 저장
    detection_data = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = map(float, det[:6])
        detection_info = {
            'class': class_names[int(cls)],
            'confidence': conf,
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        }
        detection_data.append(detection_info)

    with open(json_path, 'w') as json_file:
        json.dump(detection_data, json_file, indent=4)

    print(f"Detection saved: {image_path}, {json_path}")
    return image_path

# Flask 서버에서 GPS 데이터를 받아오는 함수
def get_gps_data_from_server():
    try:
        response = requests.get("http://127.0.0.1:5000/gps")  # Flask 서버의 GET 엔드포인트 호출
        if response.status_code == 200:
            data = response.json()
            return data.get('latitude'), data.get('longitude')
        else:
            print(f"Failed to get GPS data, status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Error fetching GPS data: {e}")
        return None, None

# 이미지 전처리 함수 (YOLOv5 모델에 맞게 조정)
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)

# 이미지 비동기 업로드 함수
def upload_images_async(images, latitude, longitude):
    for image in images:
        if os.path.exists(image):
            presigned_url, file_url = get_presigned_url()
            if not presigned_url or not file_url:
                print(f"Failed to get presigned URL for {image}")
                continue

            upload_image_to_s3(image, presigned_url)

            if os.path.exists(image):
                os.remove(image)

            register_pothole_info(file_url, latitude, longitude)
        else:
            print(f"File not found: {image}")

# 객체 검출 및 처리 함수
def detect_objects():
    save_path = os.path.join(os.getcwd(), 'detections')
    os.makedirs(save_path, exist_ok=True)

    captured_images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flask 서버에서 최신 GPS 데이터 가져오기 (주기적으로)
        latitude, longitude = get_gps_data_from_server()
        if latitude is None or longitude is None:
            latitude = 37.123  # 기본값 또는 이전 값 사용
            longitude = 127.123

        # YOLO 모델을 사용하여 객체 검출 수행
        img, ratio, dwdh = letterbox(frame, new_shape=640)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, 0.4, 0.5)

        pothole_detected = False
        detection_results = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_name = class_names[int(cls)]

                    if class_name.lower() == 'pothole':
                        pothole_detected = True
                        detection_results.append((*xyxy, conf, cls))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if pothole_detected:
            image_path = save_detection_info(frame, detection_results, save_path)
            captured_images.append(image_path)

        if len(captured_images) >= 20:
            upload_thread = threading.Thread(
                target=upload_images_async,
                args=(captured_images, latitude, longitude)
            )
            upload_thread.start()
            captured_images = []

        cv2.imshow('YOLOv5 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Flask 서버를 별도의 스레드에서 실행
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.daemon = True  # 메인 스레드가 종료되면 Flask도 종료되도록 설정
    flask_thread.start()

    # YOLO 객체 검출 함수 실행
    detect_objects()
