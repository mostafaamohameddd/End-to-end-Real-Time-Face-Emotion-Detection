import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class EmotionEngine:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            raise FileNotFoundError("XML file missing!")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def load_model(self, path):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 7)
        checkpoint = torch.load(path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model

    def process_frame(self, frame):
        output_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
        if len(faces) == 0:
            gray_eq = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(gray_eq, 1.1, 3, minSize=(30, 30))

        if len(faces) == 0:
            cv2.putText(output_frame, "NO FACE FOUND", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return output_frame, []

        detected_results = []
        for (x, y, w, h) in faces:
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            try:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.shape[0] < 10 or face_roi.shape[1] < 10: continue
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_face)
                input_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                label = CLASS_NAMES[pred.item()]
                confidence = conf.item() * 100
                
                text = f"{label}: {int(confidence)}%"
                cv2.putText(output_frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                detected_results.append({"label": label, "box": [int(x), int(y), int(w), int(h)]})

            except Exception as e:
                continue
                
        return output_frame, detected_results