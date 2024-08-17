import argparse as arg
import os
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import logging

# Set logging level
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Load the model
def load_model(path):
    return tf.keras.models.load_model(path, custom_objects={'KerasLayer': hub.KerasLayer})

# Process the image
def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

# Predict the top K classes
def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(processed_image)
    top_k_probs, top_k_classes = tf.nn.top_k(predictions, k=top_k)

    return top_k_probs.numpy()[0], top_k_classes.numpy()[0]

# Main func.
def main():
    # Create parser and add args
    parser = arg.ArgumentParser(description='Predict flower name from an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('model_path', type=str, help='Path to the trained model.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping labels to flower names.')
    
    args = parser.parse_args()

    # Load class names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = None

    # Load the model
    model = load_model(args.model_path)

    # Make predictions
    top_k_probs, top_k_classes = predict(args.image_path, model, args.top_k)

    # Display results
    print('Flower labels with corresponding probabilities:')
    for i in range(len(top_k_classes)):
        class_id = top_k_classes[i]
        if class_names:
            flower_name = class_names.get(str(class_id), f"Class {class_id}")
        else:
            flower_name = f"Class {class_id}"
        print(f"Flower Name: {flower_name}, Probability: {top_k_probs[i]}")

if __name__ == "__main__":
    main()


