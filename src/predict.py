# src/predict.py
import json
import numpy as np
from tensorflow.keras.models import load_model
from src.data_loader import load_sequence


def load_label_map(path='models/label_map.json'):
    with open(path,'r') as f:
        labels = json.load(f)
    return labels


def predict_sequence(frame_paths, model_path='models/best_model.h5', max_len=30, target_size=(64,64)):
    seq = load_sequence(frame_paths, target_size=target_size, max_len=max_len)
    seq = np.expand_dims(seq, axis=0)
    model = load_model(model_path)
    preds = model.predict(seq)
    idx = int(np.argmax(preds, axis=1)[0])
    return idx, preds[0]
