from ultralytics import YOLO
model = YOLO("yolov8m-pose.pt")
URL = "http://192.168.60.170:8080/video"
results = model(source=URL, show=True, conf=0.4, save=True)