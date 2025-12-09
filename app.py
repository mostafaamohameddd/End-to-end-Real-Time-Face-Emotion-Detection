from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from processor import EmotionEngine

# 1. تعريف التطبيق
app = Flask(__name__)
CORS(app)

# 2. إعدادات المودل
MODEL_FILE = "best_raf_db_model.pth"
engine = None

# 3. تحميل المودل
try:
    engine = EmotionEngine(MODEL_FILE)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 4. الروابط (Routes)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # استقبال الصورة
        data = request.json['image']
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # قلب الصورة للكاميرا
        frame = cv2.flip(frame, 1)

        # المعالجة
        processed_frame, results = engine.process_frame(frame)

        # ضغط الصورة لـ 40% جودة للسرعة القصوى
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
        
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image": f"data:image/jpeg;base64,{processed_base64}",
            "results": results
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        print("📥 Receiving image upload...")
        file = request.files['file']
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        # تغيير حجم الصور الكبيرة للـ Detection
        height, width = frame.shape[:2]
        if width > 800: 
            ratio = 800 / width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (800, new_height))

        # معالجة
        processed_frame, results = engine.process_frame(frame)

        # هنا بنخلي الجودة عالية شوية للصور المرفوعة (70%)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
        
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image": f"data:image/jpeg;base64,{processed_base64}",
            "results": results
        })
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # التعديل هنا:
    # host='0.0.0.0' عشان السيرفر يقبل اتصالات خارجية
    # port=7860 ده البورت القياسي لـ Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)