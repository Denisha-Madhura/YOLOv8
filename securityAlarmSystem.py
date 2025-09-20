import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email_settings import password, from_email, to_email

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(from_email, password)

def send_email(to_email, from_email, people_detected=1):
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = 'Security Alert: Person Detected'
    message.attach(MIMEText(f'Alert! {people_detected} person(s) detected by the security system.', 'plain'))
    server.sendmail(from_email, to_email, message.as_string())

class ObjectDetection:
    def __init__(self,capture_index):
        self.capture_index = capture_index
        self.email_sent = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT, thickness=3)

    def load_model(self):
        model=YOLO("yolov8m.pt")
        model.fuse()
        return model
    
    def predict(self,frame):
        results = self.model(frame)
        return results
    
    def plot_bboxes(self,frame,results):
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter for 'person' class (class ID 0)
        detections = detections[detections.class_id == 0]
        
        # Prepare labels for the filtered detections
        labels = [
            f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
            for confidence, class_id in zip(detections.confidence, detections.class_id)
        ]
        
        # Apply the annotations
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
      #  frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        # Return the annotated frame and the count of people
        people_detected_count = len(detections)
        return frame, people_detected_count
    
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        frame_count = 0

        while True:

            start_time = time()
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            frame, people_detected_count = self.plot_bboxes(frame, results)

            if people_detected_count>0:
                if not self.email_sent:
                    send_email(to_email, from_email, people_detected=people_detected_count)
                    self.email_sent = True
            else:
                self.email_sent = False

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLOv8 Detection", frame)

            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        server.quit()

detector = ObjectDetection(capture_index=0)
detector()
