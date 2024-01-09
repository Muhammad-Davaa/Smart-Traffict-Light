from ultralytics import YOLO
import cv2
import cvzone
import paho.mqtt.client as mqtt
import pandas as pd
from datetime import datetime
import os

# current_directory = os.path.dirname(os.path.abspath(__file__))
# excel_file_path = os.path.join(current_directory, 'ambulance_records.xlsx')
# if not os.path.exists(excel_file_path):
#     data = pd.DataFrame(columns=['Timestamp', 'TrafficLight', 'Duration'])
#     data.to_excel(excel_file_path, index=False)

classNames = ["ambulance"]

cap = cv2.VideoCapture("../videos/Ambulance Membelah Kemacetan 3.mp4")  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/best.pt")

mqtt_client = mqtt.Client("AmbulanceDetection")
mqtt_client.connect("mqtt.eclipseprojects.io")

data = []
start_time = None

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    ambulance_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = box.conf[0]
            cls = int(box.cls[0])

            # Filter hasil prediksi ambulans jika confidence-nya kurang dari 0.9
            if classNames[cls] == "ambulance" and conf >= 0.85:
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                ambulance_detected = True

    if ambulance_detected:
        mqtt_client.publish("ambulance_detection", "1")
        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # # Menghitung durasi dari awal hingga ambulan tidak terdeteksi lagi
        # duration = (datetime.now() - start_time).seconds if start_time else 0
        # # Simpan data ke dalam DataFrame
        # df = pd.read_excel(excel_file_path)
        # new_row = {'Timestamp': current_time, 'TrafficLight': 'Utara', 'Duration': duration}
        # df = df._append(pd.Series(new_row), ignore_index=True)
        # df.to_excel(excel_file_path, index=False)
    else:
        mqtt_client.publish("ambulance_detection", "0")

    cv2.imshow("Dava", img)
    cv2.waitKey(1)

mqtt_client.disconnect()
