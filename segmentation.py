import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  
URL = "http://192.168.60.170:8080/video"
cap = cv2.VideoCapture(URL)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results=model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Segmentation", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()