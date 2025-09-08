import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os

class AutoencoderAnomalyDetector:
    def __init__(self, input_dim, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.threshold = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        """Build autoencoder model"""
        input_layer = keras.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        self.autoencoder = keras.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        return self.autoencoder
    
    def train(self, X_train, X_val=None, epochs=50, batch_size=256):
        """Train the autoencoder"""
        if self.autoencoder is None:
            self.build_model()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, X_val_scaled) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        # Set threshold
        train_predictions = self.autoencoder.predict(X_train_scaled)
        train_mse = np.mean(np.square(X_train_scaled - train_predictions), axis=1)
        self.threshold = np.percentile(train_mse, 95)
        
        return history
    
    def predict_anomaly(self, X):
        """Predict anomalies"""
        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)
        predictions = (mse > self.threshold).astype(int)
        return predictions, mse
    
    def save_model(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.autoencoder.save(f"{filepath}_autoencoder.h5")
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'threshold': self.threshold,
                'scaler': self.scaler,
                'input_dim': self.input_dim,
                'encoding_dim': self.encoding_dim
            }, f)
    
    def load_model(self, filepath):
        """Load model"""
        self.autoencoder = keras.models.load_model(f"{filepath}_autoencoder.h5")
        
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.threshold = data['threshold']
            self.scaler = data['scaler']
            self.input_dim = data['input_dim']
            self.encoding_dim = data['encoding_dim']

def evaluate_anomaly_model(model, X_test, y_test):
    """Evaluate anomaly detection model"""
    predictions, anomaly_scores = model.predict_anomaly(X_test)
    auc = roc_auc_score(y_test, anomaly_scores)
    
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, predictions))
    
    return auc, predictions, anomaly_scores