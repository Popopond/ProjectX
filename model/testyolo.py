import os
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Using pre-trained YOLOv8 model

# Function to read labels from CSV file
def read_labels_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    print(f'Columns in {csv_file_path}: {df.columns}')  # Print columns for debugging
    labels_dict = {}
    for _, row in df.iterrows():
        image_filename = row['filename']
        class_label = row['class']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        if image_filename not in labels_dict:
            labels_dict[image_filename] = []
        labels_dict[image_filename].append([class_label, xmin, ymin, xmax, ymax])
    return labels_dict

# Function to display image with bounding boxes
def display_image_with_boxes(image_path, labels):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for label in labels:
        class_label, xmin, ymin, xmax, ymax = label

        # Set color based on class
        color = "red" if class_label == "FER" else "green"

        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        draw.text((xmin, ymin), f'Class {class_label}', fill=color)

    # Display image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Function to detect and crop eggs based on labels
def detect_and_crop_eggs(image_path, labels, output_dir):
    image_basename = os.path.basename(image_path).split('.')[0]  # Image file name without extension
    image = Image.open(image_path)  # Open image
    cropped_images = []  # Store cropped images for displaying

    for i, label in enumerate(labels):
        class_label, xmin, ymin, xmax, ymax = label

        # Crop image
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_images.append((class_label, cropped_image))  # Store the cropped image for later display

        # Determine output folder based on dataset type (train, test, valid) and class label (FER or INF)
        parent_folder = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        output_folder = os.path.join(output_dir, parent_folder, 'FER' if class_label == 'FER' else 'INF')

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Save cropped image
        output_path = os.path.join(output_folder, f'{image_basename}_{i}.jpg')
        cropped_image.save(output_path)
        print(f'Processed and saved {output_path}')
    
    return cropped_images

# Function to process all images in a folder and categorize them based on labels
def process_images_in_folder(input_folder, output_folder):
    folder_map = {
        'train': 'train_images',
        'test': 'test_images',
        'valid': 'valid_images'
    }

    for dataset_folder, images_folder_name in folder_map.items():
        folder_path = os.path.join(input_folder, dataset_folder)
        print(f'Checking folder: {folder_path}')
        if not os.path.exists(folder_path):
            print(f'Folder does not exist: {folder_path}')
            continue

        images_folder = os.path.join(folder_path, images_folder_name)
        csv_file_path = os.path.join(folder_path, '_annotations.csv')

        if not os.path.exists(images_folder):
            print(f'Images folder does not exist in {dataset_folder}: {images_folder}')
            continue

        if not os.path.exists(csv_file_path):
            print(f'_annotations.csv file does not exist in {dataset_folder}: {csv_file_path}')
            continue

        # Read labels from CSV file
        labels_dict = read_labels_from_csv(csv_file_path)

        # Display examples for both classes
        class_samples = {'FER': None, 'INF': None}
        for image_filename, labels in labels_dict.items():
            image_path = os.path.join(images_folder, image_filename)

            print(f'Processing file: {csv_file_path} and {image_path}')

            if os.path.isfile(image_path):
                if not class_samples['FER'] and any(label[0] == 'FER' for label in labels):
                    class_samples['FER'] = image_path
                if not class_samples['INF'] and any(label[0] == 'INF' for label in labels):
                    class_samples['INF'] = image_path

                # Process and crop images
                cropped_images = detect_and_crop_eggs(image_path, labels, output_folder)
                print(f'Processed {image_filename} in {dataset_folder}')

                # Display the original image with labels
                print(f'Displaying example image for class {image_filename}')
                display_image_with_boxes(image_path, labels)

                # Display the first two cropped images
                for class_label, cropped_image in cropped_images[:2]:  # Display first 2 cropped images
                    plt.imshow(cropped_image)
                    plt.title(f'Cropped Image - Class {class_label}')
                    plt.axis('off')
                    plt.show()

# Use this function with your folder paths
input_folder = r'C:/Project-Fertilizes_Egg/Dataset/Egg Tensor/'  # Main folder containing train, test, valid folders
output_folder = r'C:/Project-Fertilizes_Egg/Dataset'
process_images_in_folder(input_folder, output_folder)
