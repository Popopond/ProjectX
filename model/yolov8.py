import os
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

data_yaml_path = 'C:/Project-Fertilizes_Egg/model/data.yaml'
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"The file '{data_yaml_path}' does not exist.")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Using pre-trained YOLOv8 model

# Function to read label file
def read_labels(label_file_path):
    with open(label_file_path, 'r') as file:
        labels = [line.strip().split() for line in file]
    return labels

# Function to display image with bounding boxes
def display_image_with_boxes(image_path, labels):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    img_width, img_height = image.size

    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label)
        class_id = int(class_id)

        # Calculate bounding box coordinates
        x_center, y_center, width, height = x_center * img_width, y_center * img_height, width * img_width, height * img_height
        xmin, ymin = int(x_center - width / 2), int(y_center - height / 2)
        xmax, ymax = int(x_center + width / 2), int(y_center + height / 2)
        
        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), f'Class {class_id}', fill="red")

    # Display image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Function to detect and crop eggs based on labels
def detect_and_crop_eggs(image_path, labels, output_dir):
    image_basename = os.path.basename(image_path).split('.')[0]
    image = Image.open(image_path)

    for i, label in enumerate(labels):
        class_id, x_center, y_center, width, height = map(float, label)
        class_id = int(class_id)

        if class_id != 0:  # ตรวจสอบให้แน่ใจว่าเป็นไข่ไก่
            continue

        img_width, img_height = image.size
        x_center, y_center, width, height = x_center * img_width, y_center * img_height, width * img_width, height * img_height
        xmin, ymin = int(x_center - width / 2), int(y_center - height / 2)
        xmax, ymax = int(x_center + width / 2), int(y_center + height / 2)

        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        output_folder = os.path.join(output_dir, 'FER')  # สมมุติว่า 'FER' เป็นป้ายของไข่ไก่
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{image_basename}_{i}.jpg')
        cropped_image.save(output_path)
        print(f'Processed and saved {output_path}')


# Function to process all images in a folder and categorize them based on labels
def process_images_in_folder(input_folder, output_folder):
    for dataset_folder in ['train', 'test', 'valid']:
        folder_path = os.path.join(input_folder, dataset_folder)
        print(f'Checking folder: {folder_path}')
        if not os.path.exists(folder_path):
            print(f'Folder does not exist: {folder_path}')
            continue

        images_folder = os.path.join(folder_path, 'images')
        labels_folder = os.path.join(folder_path, 'labels')

        if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
            print(f'Images or Labels folder does not exist in {dataset_folder}')
            continue

        # Display examples for both classes
        class_samples = {'FER': None, 'INF': None}
        for filename in os.listdir(labels_folder):
            if filename.endswith('.txt'):  # Check if it's a labels file
                label_file_path = os.path.join(labels_folder, filename)
                image_filename = filename.replace('.txt', '.jpg')  # Assuming image file extension is .jpg
                image_path = os.path.join(images_folder, image_filename)

                # Check if the image file exists
                if not os.path.isfile(image_path):
                    print(f'Image file does not exist: {image_path}')
                    continue

                # Check if the label file exists
                if not os.path.isfile(label_file_path):
                    print(f'Label file does not exist: {label_file_path}')
                    continue

                print(f'Processing file: {label_file_path} and {image_path}')

                labels = read_labels(label_file_path)
                if not class_samples['FER'] and any(int(label[0]) == 0 for label in labels):
                    class_samples['FER'] = image_path
                if not class_samples['INF'] and any(int(label[0]) == 1 for label in labels):
                    class_samples['INF'] = image_path

                # Process and crop images
                detect_and_crop_eggs(image_path, labels, output_folder)
                print(f'Processed {image_filename} in {dataset_folder}')

        # Display example images for both classes
        for class_label, sample_path in class_samples.items():
            if sample_path:
                print(f'Displaying example image for class {class_label}')
                labels = read_labels(sample_path.replace('.jpg', '.txt'))
                display_image_with_boxes(sample_path, labels)
            else:
                print(f'No example image found for class {class_label}')

model.train(data='C:/Project-Fertilizes_Egg/Dataset/datayolov8/data.yaml', epochs=10, imgsz=640)

model.save('C:/Project-Fertilizes_Egg/model/good2.pt')
# Use this function with your folder paths
input_folder = r'C:/Project-Fertilizes_Egg/Dataset/datayolov8'  # Main folder containing train, test, valid folders
output_folder = r'C:/Project-Fertilizes_Egg/Dataset'
process_images_in_folder(input_folder, output_folder)
