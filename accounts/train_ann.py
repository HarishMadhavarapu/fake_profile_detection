import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ann_model.h5')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.npy')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_data.csv')  # You must provide this file

# Load your data (expects a CSV with numeric features and 'fake' as target)
df = pd.read_csv(DATA_PATH)
if 'username' in df.columns:
    df = df.drop(columns=['username'])
if 'bio' in df.columns:
    df = df.drop(columns=['bio'])

X = df.drop(columns=['fake'])
y = df['fake']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
np.save(SCALER_PATH, [scaler.mean_, scaler.scale_])

# Hyperparameters
epochs = 100
batch_size = 32
patience = 10
n_splits = 5

# Model builder for KFold
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Cross-validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
val_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f'Fold {fold+1}/{n_splits}')
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = build_model(X_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    val_pred = np.argmax(model.predict(X_val), axis=1)
    acc = accuracy_score(y_val, val_pred)
    val_scores.append(acc)
    print(f'Fold {fold+1} accuracy: {acc:.4f}')
    print(classification_report(y_val, val_pred))

print(f'Average CV accuracy: {np.mean(val_scores):.4f}')

# Retrain on full data and save final model
final_model = build_model(X_scaled.shape[1])
callbacks = [EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)]
final_model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
final_model.save(MODEL_PATH)
print('Training complete. Model and scaler saved.')