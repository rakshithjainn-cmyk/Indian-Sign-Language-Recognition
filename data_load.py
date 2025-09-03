import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(64,64)):
    X, y = [], []
    labels = os.listdir(data_dir)
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    for label in labels:
        path = os.path.join(data_dir, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label_map[label])
    
    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42), label_map

