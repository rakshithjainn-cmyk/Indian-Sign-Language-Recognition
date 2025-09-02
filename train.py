# src/train.py
import os
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from src.data_loader import prepare_dataset_from_csv
from src.model import build_cnn_lstm_model
from sklearn.model_selection import train_test_split
import numpy as np


def main(args):
    X, y, label_map = prepare_dataset_from_csv(args.data_csv, images_root=args.images_root, max_len=args.max_len, target_size=(args.h,args.w))
    num_classes = y.shape[1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=np.argmax(y,axis=1))

    model = build_cnn_lstm_model(timesteps=args.max_len, frame_shape=(args.h,args.w,3), num_classes=num_classes)
    os.makedirs('models', exist_ok=True)
    ckpt = ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ckpt, rlr, es]
    )
    # save label map
    import json
    with open('models/label_map.json', 'w') as f:
        json.dump(label_map, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--images_root', type=str, default='')
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--w', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(args)
