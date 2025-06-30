import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
import matplotlib.pyplot as plt

# Paths and constants
BASE_DIR = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'Outfit Classification')
IMAGE_SHAPE = (120, 90)
CLASS_NAMES = sorted([
    'T-Shirt/Top', 'Shirt', 'Blouse', 'Sweater', 'Jacket/Coat', 'Dress',
    'Skirt', 'Trousers/Jeans', 'Shorts', 'Shoes', 'Sandals', 'Boots',
    'Heels', 'Socks', 'Hat/Cap', 'Glasses/Sunglasses', 'Bag/Purse',
    'Scarf', 'Watch', 'Belt'
])
NUM_CLASSES = len(CLASS_NAMES)

# Load data
data = np.load(os.path.join(BASE_DIR, 'fashion_data.npz'))
images = data['images'] / 255.0
labels = data['labels']

# Train/val/test split
images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
images_val, images_test, labels_val, labels_test = train_test_split(images_temp, labels_temp, test_size=0.5, random_state=42)

# Hyperparameters
EPOCHS = 15
BATCH_SIZE = 64
HP_FILTER_SIZE = hp.HParam('filter_size', hp.Discrete([3, 5]))
HP_FILTER_NUM = hp.HParam('filters_number', hp.Discrete([32, 64]))
METRIC_ACCURACY = 'accuracy'
FILE_WRITER_DIR = os.path.join(BASE_DIR, 'Logs', 'Model 1', 'hparam_tuning')

with tf.summary.create_file_writer(FILE_WRITER_DIR).as_default():
    hp.hparams_config(
        hparams=[HP_FILTER_SIZE, HP_FILTER_NUM],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

def build_model(hparams):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(hparams[HP_FILTER_NUM], hparams[HP_FILTER_SIZE], activation='relu', input_shape=(*IMAGE_SHAPE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(hparams[HP_FILTER_NUM], 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

def train_test_model(hparams, session_num):
    model = build_model(hparams)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    log_dir = os.path.join(BASE_DIR, 'Logs', 'Model 1', 'fit', f"run-{session_num}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    model.fit(images_train, labels_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(images_val, labels_val),
              callbacks=[tensorboard_callback, early_stopping],
              verbose=2)

    accuracy = model.evaluate(images_val, labels_val, verbose=0)[1]
    model.save(os.path.join(BASE_DIR, 'saved_models', 'Model 1', f'Run-{session_num}.h5'))
    return accuracy

def run(log_dir, hparams, session_num):
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)
        acc = train_test_model(hparams, session_num)
        tf.summary.scalar(METRIC_ACCURACY, acc, step=1)

# Run all sessions
session_num = 1
for filter_size in HP_FILTER_SIZE.domain.values:
    for filter_num in HP_FILTER_NUM.domain.values:
        hparams = {HP_FILTER_SIZE: filter_size, HP_FILTER_NUM: filter_num}
        run(FILE_WRITER_DIR + f"/run-{session_num}", hparams, session_num)
        session_num += 1

# Final evaluation
best_model_path = os.path.join(BASE_DIR, 'saved_models', 'Model 1', 'Run-1.h5')
model = tf.keras.models.load_model(best_model_path)
test_loss, test_acc = model.evaluate(images_test, labels_test)
print(f"Test Accuracy: {test_acc*100:.2f}%, Test Loss: {test_loss:.4f}")
