import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ann_model.h5')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.npy')

# Cache model and scaler at module level
_model = None
_scaler = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        scaler = StandardScaler()
        if os.path.exists(SCALER_PATH):
            scaler.mean_, scaler.scale_ = np.load(SCALER_PATH, allow_pickle=True)
        _scaler = scaler
    return _scaler

class InstagramANNPredictor:
    def __init__(self):
        self.model = get_model()
        self.scaler = get_scaler()

    def preprocess(self, df):
        if 'fake' in df.columns:
            df = df.drop(columns=['fake'])
        return self.scaler.transform(df)

    def predict(self, df):
        X = self.preprocess(df)
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1), preds

def save_scaler(scaler):
    np.save(SCALER_PATH, [scaler.mean_, scaler.scale_])