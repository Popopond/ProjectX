from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import io
import base64
import sqlite3
from database import create_database
import matplotlib.pyplot as plt
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Create database when app starts
create_database()

# Load YOLOv8 and egg classifier models
yolov8_model = YOLO("C:/Project-Fertilizes_Egg/model/good2.pt")
classifier_model = tf.keras.models.load_model("C:/Project-Fertilizes_Egg/model/best_model.keras")

def preprocess_image(image):
    """Preprocess the input image for classification."""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def get_db_connection():
    """Create and return a new database connection."""
    conn = sqlite3.connect('C:\\Project-Fertilizes_Egg\\flask_api\\egg_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

def execute_query(query, params=()):
    """Execute a database query and return the results."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

def get_membership_info(user_id):
    """Retrieve membership status and check count for a given user ID."""
    results = execute_query("SELECT membership_status, check_count FROM users WHERE user_id=?", (user_id,))
    return (results[0]['membership_status'], results[0]['check_count']) if results else (None, 0)

def update_check_count(user_id):
    """Increment the check count for the specified user."""
    execute_query("UPDATE users SET check_count = check_count + 1 WHERE user_id=?", (user_id,))

def update_user_status_to_member(user_id):
    """Upgrade user membership status to 'member'."""
    execute_query("UPDATE users SET membership_status='member' WHERE user_id=?", (user_id,))

def summarize_and_analyze(user_id):
    """Summarize fertilized and unfertilized egg data for the user."""
    results = execute_query('''
    SELECT is_fertilized, COUNT(*) as count
    FROM image_submissions
    WHERE user_id = ?
    GROUP BY is_fertilized
    ''', (user_id,))
    
    fertilized_count = sum(row['count'] for row in results if row['is_fertilized'] == 1)
    unfertilized_count = sum(row['count'] for row in results if row['is_fertilized'] == 0)

    labels = ['Fertilized', 'Unfertilized']
    sizes = [fertilized_count, unfertilized_count]
    colors = ['#ff9999', '#66b3ff']
    
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()

    return {
        'fertilized_count': fertilized_count,
        'unfertilized_count': unfertilized_count,
        'image_base64': img_base64
    }

#ของใหม่
def generate_pdf_report(user_id, summary_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, f"Egg Analysis Report for User ID: {user_id}")
    
    # เพิ่มข้อมูลประวัติ
    c.drawString(100, 700, f"Fertilized Count: {summary_data['fertilized_count']}")
    c.drawString(100, 680, f"Unfertilized Count: {summary_data['unfertilized_count']}")
    
    # เพิ่มกราฟ
    img_data = base64.b64decode(summary_data['image_base64'])
    img_io = io.BytesIO(img_data)
    img = ImageReader(img_io)
    c.drawImage(img, 100, 450, width=400, height=200)
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer

def new_user(username, lastname, birthdate, email, line_login):
    """Register a new user in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if the user already exists
        existing_user = execute_query("SELECT * FROM users WHERE line_login=?", (line_login,))
        
        if existing_user:
            # If user already exists, return their information
            return existing_user[0]['user_id']
        
        cursor.execute(
            "INSERT INTO users (username, lastname, birthdate, email, line_login, membership_status, check_count) VALUES (?, ?, ?, ?, ?, 'non-member', 0)",
            (username, lastname, birthdate, email, line_login)
        )
        conn.commit()
        
        return cursor.lastrowid  # Return the newly created user's ID

def get_user_id_from_line_login(line_login):
    """Retrieve user ID from line_login."""
    results = execute_query("SELECT user_id FROM users WHERE line_login=?", (line_login,))
    return results[0]['user_id'] if results else None

@app.route('/summarize/<user_id>', methods=['GET'])
def summarize(user_id):
    summary_data = summarize_and_analyze(user_id)
    return jsonify(summary_data)

#ของใหม่
@app.route('/generate_report/<user_id>', methods=['GET'])
def generate_report(user_id):
    """Generate a PDF report with the egg analysis summary."""
    try:
        summary_data = summarize_and_analyze(user_id)
        pdf_buffer = generate_pdf_report(user_id, summary_data)
        
        response = make_response(pdf_buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=report_{user_id}.pdf'
        return response
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_line_login', methods=['POST'])
def update_line_login():
    """Update the LINE login ID for a user."""
    try:
        data = request.json
        old_line_login = data['old_line_login']
        new_line_login = data['new_line_login']
        
        execute_query("UPDATE users SET line_login=? WHERE line_login=?", (new_line_login, old_line_login))
        return jsonify({'message': 'Line ID updated successfully'}), 200
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    """Register a new user or update existing user information."""
    try:
        data = request.json
        required_keys = ['username', 'birthdate', 'email', 'line_login']
        
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing required parameter: {key}'}), 400

        line_login = data['line_login']
        existing_user = execute_query("SELECT * FROM users WHERE line_login=?", (line_login,))

        if existing_user:
            # Update existing user info and change status to 'member'
            with get_db_connection() as conn:
                cursor = conn.cursor()
                # Print existing user information before update
                print(f"Existing user before update: {existing_user}")

                cursor.execute(
                    "UPDATE users SET username=?, lastname=?, birthdate=?, email=?, membership_status='member' WHERE line_login=?",
                    (data['username'], data.get('lastname', ''), data['birthdate'], data['email'], line_login)
                )
                conn.commit()

                # Verify the update
                updated_user = execute_query("SELECT * FROM users WHERE line_login=?", (line_login,))
                print(f"Updated user information: {updated_user}")

            return jsonify({'message': 'User information updated successfully and membership upgraded to member'}), 200
        else:
            # If user does not exist, return an error message
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/check_membership/<line_login>', methods=['GET'])
def check_membership(line_login):
    """Check membership status based on line_login."""
    user_id = get_user_id_from_line_login(line_login)
    
    if user_id is None:
        return jsonify({'error': 'User not found'}), 404

    membership_status, check_count = get_membership_info(user_id)
    
    if membership_status == 'non-member' and check_count >= 5:
        return jsonify({'message': 'Please subscribe to become a member'}), 200
    elif membership_status == 'member':
        return jsonify({'message': 'You are a member'}), 200
    else:
        return jsonify({'message': 'You have free access'}), 200



def save_image_submission(user_id, image, is_fertilized, accuracy):
    """Save the image submission to the database and update egg counts."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Save the image submission
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        cursor.execute(
            "INSERT INTO image_submissions (user_id, image_path, is_fertilized, accuracy) VALUES (?, ?, ?, ?)",
            (user_id, img_data, is_fertilized, accuracy)
        )
        
        # Update egg counts based on whether the egg is fertilized or unfertilized
        if is_fertilized == 1:
            cursor.execute(
                "UPDATE egg_counts SET fertilized_count = fertilized_count + 1, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,)
            )
        else:
            cursor.execute(
                "UPDATE egg_counts SET unfertilized_count = unfertilized_count + 1, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,)
            )

        # If the user doesn't have an entry in the egg_counts table, create one
        cursor.execute(
            "INSERT INTO egg_counts (user_id, fertilized_count, unfertilized_count) SELECT ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM egg_counts WHERE user_id = ?)",
            (user_id, 1 if is_fertilized == 1 else 0, 0 if is_fertilized == 1 else 1, user_id)
        )

        conn.commit()

@app.route('/update_membership', methods=['POST'])
def update_membership():
    """Update the membership status to 'member' for an existing user."""
    try:
        data = request.json
        required_keys = ['user_id']
        
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing required parameter: {key}'}), 400

        user_id = data['user_id']

        # Check if the user exists
        existing_user = execute_query("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        
        if not existing_user:
            return jsonify({'error': 'User not found'}), 404
        
        # Update the membership status to 'member'
        update_user_status_to_member(user_id)
        
        return jsonify({'message': 'Membership status updated to member successfully'}), 200
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    user_id = request.form['userId']  # รับ user_id จาก request form
    image = Image.open(request.files['image'].stream)
    draw = ImageDraw.Draw(image)

    membership_status, check_count = get_membership_info(user_id)

    if membership_status is None:
        username = f"User_{user_id}"
        user_id = new_user(username=username, lastname="", birthdate="", email="", line_login=user_id)
        membership_status = 'non-member'
        check_count = 0

    if membership_status == 'non-member' and check_count >= 5:
        return jsonify({'message': 'You have reached the limit of 5 image checks.'}), 403

    results = yolov8_model(image)
    predictions = []
    cropped_images = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image.crop((x1, y1, x2, y2))
            preprocessed_image = preprocess_image(cropped_image)
            prediction = classifier_model.predict(preprocessed_image)
            is_fertilized = 'fertilized' if prediction[0][0] > 0.5 else 'unfertilized'
            accuracy = float(prediction[0][0]) if is_fertilized == 'fertilized' else float(1 - prediction[0][0])
            predictions.append((is_fertilized, round(accuracy, 3)))
            cropped_images.append(cropped_image)
            label = f"{is_fertilized} ({accuracy:.3f})"
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, y1 - 10), label, fill="red")

    for cropped_image, (is_fertilized, accuracy) in zip(cropped_images, predictions):
        save_image_submission(user_id, cropped_image, 1 if is_fertilized == 'fertilized' else 0, accuracy)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    update_check_count(user_id)

    membership_status, check_count = get_membership_info(user_id)
    if membership_status == 'non-member' and check_count >= 5:
        return jsonify({'message': 'You have reached the limit of 5 image checks.'}), 403

    response = {
        'predictions': [{'status': is_fertilized, 'accuracy': f"{accuracy:.3f}"} for is_fertilized, accuracy in predictions],
        'annotated_image': img_base64
    }

    return jsonify(response), 200






if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
