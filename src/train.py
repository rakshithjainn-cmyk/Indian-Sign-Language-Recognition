import numpy as np
from src.data_loader import load_data
from src.model import create_cnn
import joblib
import os

def train(data_dir='data/signs'):
    os.makedirs('models', exist_ok=True)
    (X_train, X_test, y_train, y_test), label_map = load_data(data_dir)
    model = create_cnn((64, 64, 1), num_classes=len(label_map))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save('models/sign_model.h5')
    joblib.dump(label_map, 'models/label_map.pkl')
    print(" Model training complete. Saved to models/")

if __name__ == '__main__':
    train()
