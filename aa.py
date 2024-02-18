import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Function to define YOLOv7 model architecture
def yolov7(input_shape, num_classes, num_boxes=3):
    inputs = tf.keras.Input(shape=input_shape)

    # Backbone
    x = Conv2D(64, 3, strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(1024, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(1024, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Head
    x = Conv2D(1024, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(num_boxes * (5 + num_classes), 1, strides=(1, 1), padding='same')(x)
    predictions = Reshape((13, 13, num_boxes, 5 + num_classes))(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

# Function to parse YOLO format labels
def parse_yolo_label(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
        label = [float(x) for x in lines[0].strip().split()]
    return label

# Custom data generator for training
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, label_dir, batch_size, image_size=(416, 416), num_classes=1):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.image_paths = sorted(os.listdir(image_dir))
        self.label_paths = sorted(os.listdir(label_dir))
        self.num_samples = len(self.image_paths)

    def __len__(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_paths = self.label_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for image_path, label_path in zip(batch_image_paths, batch_label_paths):
            image = cv2.imread(os.path.join(self.image_dir, image_path))
            image = cv2.resize(image, self.image_size)
            batch_images.append(image)
            label = parse_yolo_label(os.path.join(self.label_dir, label_path))
            batch_labels.append(label)
        return np.array(batch_images), np.array(batch_labels)

# Define paths to training data
train_img_dir = "C:\\Users\\My pc\\Downloads\\collegeproject-master\\collegeproject-master\\train\\images"
train_label_dir = "C:\\Users\\My pc\\Downloads\\collegeproject-master\\collegeproject-master\\train\\labels"

# Define hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = (416, 416)
NUM_CLASSES = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Create custom data generator for training
train_generator = CustomDataGenerator(train_img_dir, train_label_dir, BATCH_SIZE, IMAGE_SIZE, NUM_CLASSES)

# Create and compile YOLOv7 model
model = yolov7(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

# Train the model
model.fit(train_generator, epochs=NUM_EPOCHS)

# Calculate predictions on the training set
train_predictions = model.predict(train_generator)

# Flatten the predictions and ground truth labels
train_labels = np.concatenate([y for x, y in train_generator], axis=0)

# Calculate accuracy
train_accuracy = accuracy_score(np.argmax(train_labels, axis=-1).flatten(), np.argmax(train_predictions, axis=-1).flatten())

print("Train Accuracy:", train_accuracy)

