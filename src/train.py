# src/train.py
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from src.data_loader import prepare_dataset_from_csv
from src.model import build_cnn_lstm_model
from sklearn.model_selection import train_test_split

# Also set TensorFlow threading limits at runtime
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

print(" TensorFlow version:", tf.__version__)
print(" Available devices:", tf.config.list_physical_devices())
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f" Using GPU: {gpu_devices}")
else:
    print(" No GPU found, running on CPU.")


def main(args):
    print(" Loading dataset...")
    X, y, label_map = prepare_dataset_from_csv(
        args.data_csv,
        images_root=args.images_root,
        max_len=args.max_len,
        target_size=(args.h, args.w)
    )
    num_classes = y.shape[1]
    print(f"Samples: {X.shape[0]}, Timesteps: {X.shape[1]}, Classes: {num_classes}")

    # stratify expects class indices
    y_indices = np.argmax(y, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y_indices,
        shuffle=True
    )

    model = build_cnn_lstm_model(
        timesteps=args.max_len,
        frame_shape=(args.h, args.w, 3),
        num_classes=num_classes,
        learning_rate=args.lr
    )

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, 'best_model.h5')
    ckpt = ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

    print(" Starting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ckpt, rlr, es]
    )

    # Save label map (index -> label)
    label_map_path = os.path.join(args.save_dir, 'label_map.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f" Saved model to: {ckpt_path}")
    print(f" Saved label map to: {label_map_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--images_root', type=str, default='')
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--w', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
