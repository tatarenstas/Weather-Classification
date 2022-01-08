from tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

img_width=256
img_height=256
weather = ['Cloudy','Foggy','Rainy','Shine','Sunrise']
model = keras.models.load_model('model.h5')

def preprocess_image(path):
    img = load_img(path, target_size = (img_height, img_width))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.0
    return a

image_path = '/content/drive/MyDrive/Colab Notebooks/dataset/test/sunrise_7.jpg'
image = preprocess_image(image_path)
predict = model.predict(image)
result = np.argmax(predict)
print(weather[result])
