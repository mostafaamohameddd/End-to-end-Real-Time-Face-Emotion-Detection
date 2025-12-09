import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import urllib.request

# ---  ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----
def load_model(model_path):
    print(f"Checking model path: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {os.path.abspath(model_path)}")
        return None
        
    print("Loading model architecture...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    
    print("Loading weights...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

def start_webcam(model):

    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print("Downloading face detection file...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)

    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ==========================================
        # ااً (Mirror)
        frame = cv2.flip(frame, 1) 
        # ==========================================

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_roi = frame[y:y+h, x:x+w]
            try:
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_face)
                input_tensor = data_transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                label = class_names[pred.item()]
                confidence = conf.item() * 100
                
                label_text = f"{label}: {confidence:.1f}%"
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                pass

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model('best_raf_db_model.pth')
    if model:
        start_webcam(model)