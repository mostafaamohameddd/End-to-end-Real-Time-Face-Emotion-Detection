# Real-Time Facial Emotion Detection AI  
**End-to-End Deep Learning System for Live & Static Emotion Analysis**

A production-grade computer vision application for real-time facial emotion recognition using a fine-tuned **ResNet50 (PyTorch)** model trained on the **RAF-DB** dataset.  
The system supports **live webcam inference** and **static image analysis**, delivered through a modern web interface and deployed using **Docker on Hugging Face Spaces** for high availability.

<video src="./assets/mo_test.mp4" controls="controls" width="100%">
  Your browser does not support the video tag.
</video>


---

## Table of Contents
1. Overview  
2. Live Demo  
3. Key Features  
4. System Architecture  
5. Model & Dataset  
6. Tech Stack  
7. Installation & Local Usage  
8. Docker Deployment  
9. Hugging Face Deployment
10. 10. Project Structure
11. Performance Considerations  
12. Limitations & Future Work  
13. License  
14. Author  

---

## Overview
This project demonstrates a full **end-to-end AI pipeline**, covering:
- Model training and fine-tuning
- Real-time computer vision inference
- Backend APIs for low-latency prediction
- Modern frontend UI
- Cloud deployment with containerization

It is designed to reflect **industry-level engineering practices** and can be extended into production systems such as emotion-aware assistants, mental health tools, or human-computer interaction platforms.

---

## Live Demo
The application is publicly deployed and fully functional:

🔗 **Hugging Face Spaces (Live Webcam + Image Upload)**  
https://huggingface.co/spaces/mostafaamohamedd14/Face-Emotion-Detector-Live

---

## Key Features
- **Real-Time Emotion Detection**  
  Low-latency inference from live webcam streams.

- **Static Image Analysis**  
  Upload images for fast and accurate emotion classification.

- **Deep Learning Powered**  
  Fine-tuned ResNet50 trained on RAF-DB facial expression dataset.

- **Adaptive Face Processing**  
  Handles varying lighting conditions, face sizes, and positions.

- **Production-Ready Deployment**  
  Fully containerized using Docker and deployed on Hugging Face Spaces.

- **Modern UI**  
  Clean, responsive frontend built with Tailwind CSS.

---

## System Architecture
The system follows a modular architecture:

1. **Input Layer**
   - Webcam stream (browser-based)
   - Uploaded static images

2. **Preprocessing**
   - Face detection using Haar Cascades (OpenCV)
   - Face cropping, resizing, normalization

3. **Inference Engine**
   - PyTorch ResNet50 model
   - GPU/CPU-agnostic inference

4. **Backend**
   - Flask REST server
   - Optimized request handling for real-time predictions

5. **Frontend**
   - HTML5 + Tailwind CSS + JavaScript
   - Live video feed and prediction rendering

6. **Deployment**
   - Docker container
   - Hugging Face Spaces runtime

---

## Model & Dataset
### Model
- **Architecture:** ResNet50  
- **Framework:** PyTorch  
- **Strategy:** Transfer learning + fine-tuning  
- **Output:** Multi-class emotion classification

### Dataset
- **RAF-DB (Real-world Affective Faces Database)**
- Contains real-world facial expressions under varying conditions.
- Commonly used in academic and industrial research.

---

## Tech Stack
### Core
- Python 3.9

### AI / ML
- PyTorch
- torchvision
- NumPy

### Computer Vision
- OpenCV
- Haar Cascade Face Detection

### Backend
- Flask

### Frontend
- HTML5
- Tailwind CSS
- JavaScript

### Deployment
- Docker
- Hugging Face Spaces

---

## Installation & Local Usage

### 1. Clone the Repository
```bash
git clone https://github.com/mostafaamohameddd/Emotion-Detection.git
cd Emotion-Detection


### 2. Create Virtual Environment (Recommended)

python -m venv venv

source venv/bin/activate  # Linux / macOS

venv\Scripts\activate     # Windows




### 3. Install Dependencies

pip install -r requirements.txt

python app.py

http://127.0.0.1:5000




### 4. Docker Deployment
docker build -t emotion-detector .

docker run -p 7860:7860 emotion-detector




### 5.Hugging Face Deployment
This project is optimized for Hugging Face Spaces deployment:

Docker-based runtime

Automatic rebuilds on push

Public access with webcam permissions

Live deployment:
https://huggingface.co/spaces/mostafaamohamedd14/Face-Emotion-Detector-Live


### Project Structure
├── app.py                     # Flask server (entry point)
├── processor.py               # Emotion inference & preprocessing logic
├── web_cam.py                 # Local webcam testing script
├── Face_emotion_detection.ipynb# Model training notebook
├── best_raf_db_model.pth      # Trained PyTorch model
├── haarcascade_*.xml          # Haar cascade face detector
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Dependencies
└── templates/
    └── index.html             # Frontend UI


###Performance Considerations
Real-time inference optimized for CPU environments.

Face detection resolution dynamically adjusted.

Can be extended to GPU inference for higher FPS.

Scalable backend via container orchestration


### Limitations & Future Work
Replace Haar Cascades with deep face detectors (e.g., RetinaFace).

Add face tracking to reduce redundant detections.

Integrate temporal modeling (LSTM/Transformer) for emotion smoothing.

Expand emotion classes and dataset diversity.

Add REST API endpoints for external integrations.


### License
This project is licensed under the MIT License.


### Author
Mostafa Mohamed
Deployment: https://huggingface.co/spaces/mostafaamohamedd14/Face-Emotion-Detector-Live
