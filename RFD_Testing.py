# Including Essential Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Training dataset
train_path = 'real_and_fake_face'

# Defining Categories
categories = ["Fake", "Real"]

# Loading the Saved Model
from tensorflow.keras.models import load_model
model = load_model('RFD_0303.h5')

def load_image(filename, show = True):
    img = cv2.imread(filename)
    # cv2.imshow('image', img)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis = 0)
    img = img / 255.
    if show:
        plt.imshow(img[0])
        plt.axis('off')
        plt.show()
    return img

filename = 'real (1).jpg'
pred=model.predict([load_image(filename)])
pred
np.argmax(pred)
categories[np.argmax(pred)]