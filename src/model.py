# src/model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam


def build_cnn_frame(input_shape=(64,64,3)):
    from tensorflow.keras import layers
    inp = Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    model = Model(inp, x, name='cnn_frame')
    return model


def build_cnn_lstm_model(timesteps=30, frame_shape=(64,64,3), num_classes=10):
    frame_model = build_cnn_frame(input_shape=frame_shape)
    seq_in = Input(shape=(timesteps, frame_shape[0], frame_shape[1], frame_shape[2]))
    x = TimeDistributed(frame_model)(seq_in)
    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(seq_in, out)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
