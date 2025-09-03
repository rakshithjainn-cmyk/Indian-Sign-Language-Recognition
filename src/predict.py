import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model

model = load_model('models/sign_model.h5')
label_map = joblib.load('models/label_map.pkl')
inv_label_map = {v: k for k, v in label_map.items()}

def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    img = img.reshape(1,64,64,1) / 255.0
    pred = model.predict(img)
    label = inv_label_map[np.argmax(pred)]
    return label
