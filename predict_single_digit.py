import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("../models/digit_cnn_model.h5")

img_path = "../data/sample_digit.png"   # black digit on white background
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = 255 - img   # invert colors if needed
img = img.reshape(1, 28, 28, 1).astype("float32") / 255.0

pred = np.argmax(model.predict(img))
print("Predicted Digit:", pred)
