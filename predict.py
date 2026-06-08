from tensorflow.keras.models import load_model
import cv2

model = load_model("models/mnist_cnn.keras")

img = cv2.imread("images/test.png")

print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray.shape)
gray = cv2.resize(gray, (28, 28))

print(gray.shape)
gray = 255 - gray
print(gray.shape)
import matplotlib.pyplot as plt

plt.imshow(gray, cmap='gray')
plt.show()
gray = gray / 255.0

print(gray.min())
print(gray.max())
gray = gray.reshape(1, 28, 28, 1)

print(gray.shape)
import numpy as np

prediction = model.predict(gray)

print(prediction)

digit = np.argmax(prediction)

confidence = np.max(prediction) * 100

print("Predicted Digit:", digit)
print("Confidence:", confidence, "%")