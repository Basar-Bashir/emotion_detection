import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("fer2013.csv")

pixels = data['pixels'].tolist()
faces = []

for pixel in pixels:
    face = np.array(pixel.split(), dtype='float32')
    face = face.reshape(48, 48, 1)
    faces.append(face)

faces = np.array(faces) / 255.0
labels = pd.get_dummies(data['emotion']).values

X_train, X_test, y_train, y_test = train_test_split(
    faces, labels, test_size=0.2, random_state=42
)

np.save("dataset/X_train.npy", X_train)
np.save("dataset/X_test.npy", X_test)
np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)

print("Preprocessing complete.")
