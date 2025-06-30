import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Update this to the correct model path
MODEL_DIR = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'Outfit Classification', 'saved_models', 'Model 1', 'Run-1.h5')

# Classes (sorted based on labels used in training)
class_names = sorted([
    'T-Shirt/Top', 'Shirt', 'Blouse', 'Sweater', 'Jacket/Coat', 'Dress',
    'Skirt', 'Trousers/Jeans', 'Shorts', 'Shoes', 'Sandals', 'Boots',
    'Heels', 'Socks', 'Hat/Cap', 'Glasses/Sunglasses', 'Bag/Purse',
    'Scarf', 'Watch', 'Belt'
])

def classify_image(image_path, model_path=MODEL_DIR):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(image_path, target_size=(120, 90))  # Match model input shape
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_array)[0]
    predicted_class = np.argmax(probs)
    predicted_label = class_names[predicted_class]
    confidence = probs[predicted_class] * 100

    print(f"Predicted: {predicted_label} ({confidence:.2f}%)")
    return predicted_label, confidence
