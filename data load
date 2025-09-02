# src/data_loader.py
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def load_image(path, target_size=(64,64)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32')/255.0
    return img


def load_sequence(sequence_paths, target_size=(64,64), max_len=30):
    seq = []
    for p in sequence_paths[:max_len]:
        seq.append(load_image(p, target_size))
    # pad if shorter
    while len(seq) < max_len:
        seq.append(np.zeros((target_size[0], target_size[1], 3), dtype='float32'))
    return np.stack(seq)


def prepare_dataset_from_csv(csv_path, images_root='', max_len=30, target_size=(64,64)):
    """
    CSV format: sequence_id,frame_paths(comma-separated),label
    Or: filepath,label for single image classification dataset
    Return: X (num_samples, timesteps, h, w, c), y (num_samples, )
    """
    df = pd.read_csv(csv_path)
    X = []
    y = []
    for _, row in df.iterrows():
        if 'frame_paths' in row and pd.notna(row['frame_paths']):
            frames = row['frame_paths'].split(';')
            frames = [os.path.join(images_root, f.strip()) if images_root else f.strip() for f in frames]
            seq = load_sequence(frames, target_size=target_size, max_len=max_len)
            X.append(seq)
            y.append(row['label'])
        elif 'filepath' in row and pd.notna(row['filepath']):
            p = os.path.join(images_root, row['filepath']) if images_root else row['filepath']
            img = load_image(p, target_size=target_size)
            # convert to sequence length 1 + padding
            seq = np.concatenate([np.expand_dims(img,0), np.zeros((max_len-1, target_size[0], target_size[1],3))], axis=0)
            X.append(seq)
            y.append(row['label'])
        else:
            raise ValueError('CSV must contain either frame_paths or filepath columns')
    X = np.array(X)
    labels, uniques = pd.factorize(y)
    y = to_categorical(labels)
    return X, y, list(uniques)
