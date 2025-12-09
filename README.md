#  Real-Time Emotion Detection AI

A professional computer vision project that detects human emotions in real-time using **Deep Learning (ResNet50)** and **Flask**.

##  Live Demo
Check out the live running application on Hugging Face:
 **[Click Here to Try the App](https://huggingface.co/spaces/mostafaamohamedd14/Face-Emotion-Detector-Live)**

---

##  Tech Stack
* **Core:** Python 3.9
* **AI Model:** ResNet50 (PyTorch) trained on RAF-DB dataset.
* **Web Framework:** Flask.
* **Computer Vision:** OpenCV (Haar Cascades).
* **Frontend:** HTML5, Tailwind CSS, JavaScript.
* **Deployment:** Docker & Hugging Face Spaces.

---

## ✨ Features
* **Real-Time Detection:** Smooth emotion recognition from webcam feed.
* **Upload Analysis:** Smart resizing and analysis for uploaded images.
* **Adaptive Processing:** Auto-adjusts for lighting and face size.
* **Modern UI:** Cyberpunk/Glassmorphism design style.

---

## 📦 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/اسمك/Emotion-Detection.git](https://github.com/mostafaamohameddd/Emotion-Detection.git)
   cd Emotion-Detection
   
   
#Install dependencies:

pip install -r requirements.txt

python app.py

#License
This project is licensed under the MIT License - see the LICENSE file for details.

#Project Structure
├── app.py                   # Flask Server (Entry Point)
├── processor.py             # Emotion Recognition Logic & Smart Detection
├── web_cam.py               # Local script for testing webcam without browser
├── Face_emotion_detection.ipynb # Model Training Notebook
├── best_raf_db_model.pth    # Trained PyTorch Model
├── haarcascade_...xml       # Face Detection Model
├── Dockerfile               # Container configuration for Cloud Deployment
├── requirements.txt         # Project Dependencies
└── templates/
    └── index.html           # The Modern UI

