from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import io
import os
import base64
from datetime import datetime
import sqlite3
app = Flask(__name__)

# โหลดโมเดล YOLOv8 และโมเดลการคัดแยกไข่
yolov8_model = YOLO("C:/Project-Fertilizes_Egg/model/good2.pt")
classifier_model = tf.keras.models.load_model("C:/Project-Fertilizes_Egg/model/best_model.keras")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# สร้างการเชื่อมต่อกับฐานข้อมูล SQLite
def get_db_connection():
    conn = sqlite3.connect('egg_detection.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = Image.open(request.files['image'].stream)
    draw = ImageDraw.Draw(image)

    # ตรวจจับไข่และครอปภาพ
    results = yolov8_model(image)
    detections = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0], 'boxes') else []

    predictions = []
    cropped_images = []

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection[:4])
        crop_img = image.crop((x1, y1, x2, y2))

        # ประมวลผลภาพที่ครอป
        processed_img = preprocess_image(crop_img)
        
        # พยากรณ์ผลลัพธ์ด้วย classifier_model
        prediction = classifier_model.predict(processed_img)
        print(f"Prediction for crop {i+1}: {prediction}")  # แสดงผลการพยากรณ์
        
        label = 'Fertilized' if prediction > 0.5 else 'Unfertilized'
        print(f"Label assigned for crop {i+1}: {label}")  # แสดงผล label ที่ถูกกำหนด

        predictions.append(label)

        # วาดกรอบและชื่อผลลัพธ์บนภาพต้นฉบับ
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")

        # บันทึกภาพที่ครอปไว้ในฐานข้อมูล
        # save_image_to_db(crop_img, label, user_id='example_user_id')  # ระบุ user_id

        # เข้ารหัสภาพเป็น base64 เพื่อส่งกลับไปยัง LINE
        img_byte_arr = io.BytesIO()
        crop_img.save(img_byte_arr, format='JPEG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        cropped_images.append({
            'image': img_base64,
            'label': label
        })

    # บันทึกภาพต้นฉบับที่ถูกตรวจจับและครอปในฐานข้อมูล
    # save_image_to_db(image, 'detected_image', user_id='example_user_id')  # ระบุ user_id

    # เข้ารหัสภาพต้นฉบับเป็น base64 เพื่อส่งกลับไปยัง LINE
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return jsonify({
        'predictions': predictions,
        'image': img_base64
    })


if __name__ == '__main__':
    app.run(debug=True)