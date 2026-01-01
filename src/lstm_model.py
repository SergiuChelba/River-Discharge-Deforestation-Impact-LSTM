import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np

class FloodLSTMModel:
    def __init__(self, input_shape, units=64, dropout_rate=0.2, learning_rate=0.001):
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_architecture()
        
    def _build_architecture(self):
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(self.units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath='models/final_model.keras'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    @staticmethod
    def load(filepath='models/final_model.keras'):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found at {filepath}")
        return load_model(filepath)