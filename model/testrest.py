import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Path to your saved model
model_path = 'C:/Project-Fertilizes_Egg/model/best_model.keras'

# Load the model
model = tf.keras.models.load_model(model_path)

# Path to new image
image_path = 'C:\Project-Fertilizes_Egg\Dataset\datayolov8\crop/test\FER\images-50-_jpg_0.jpg'

# Parameters
image_size = (224, 224)  # Adjust based on your model's input size

# Prepare the image
def prepare_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict the class of the new image
def predict_image(model, image_path, image_size):
    img_array = prepare_image(image_path, image_size)
    prediction_prob = model.predict(img_array)[0, 0]
    predicted_class = 'FER' if prediction_prob > 0.5 else 'INF'
    return predicted_class, prediction_prob

# Get prediction
predicted_class, prediction_prob = predict_image(model, image_path, image_size)

# Display the image and prediction
def display_image_and_prediction(image_path, predicted_class, prediction_prob):
    img = load_img(image_path, target_size=image_size)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class}\nProbability: {prediction_prob:.2f}')
    plt.show()

# Show the result
display_image_and_prediction(image_path, predicted_class, prediction_prob)
