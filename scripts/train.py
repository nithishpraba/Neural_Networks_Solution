import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.logger import get_logger

logger = get_logger(__name__)

def load_and_preprocess_data():
    # Update the file name if needed (ensure Admission(in).csv is in the data folder)
    data_path = os.path.join("data", "Admission(in).csv")
    try:
        df = pd.read_csv(data_path)
        logger.info("Dataset loaded. Shape: %s", df.shape)
    except Exception as e:
        logger.error("Error loading data from %s: %s", data_path, e)
        return None, None

    # Set target column and drop the identifier
    target_column = "Admit_Chance"
    if target_column not in df.columns:
        logger.error("Target column '%s' not found. Available columns: %s", target_column, df.columns.tolist())
        return None, None

    # Drop the Serial_No column (assuming it's an identifier)
    if "Serial_No" in df.columns:
        df = df.drop("Serial_No", axis=1)
    
    # Split features and target
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler (for consistent inference)
    os.makedirs("models", exist_ok=True)
    scaler_path = os.path.join("models", "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    return X_scaled, y

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    # For regression, we use linear activation
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model():
    X, y = load_and_preprocess_data()
    if X is None or y is None:
        return None

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, 
              callbacks=[early_stop], verbose=1)
    return model

def main():
    model = train_model()
    if model is not None:
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.h5")
        model.save(model_path)
        logger.info("Model saved to %s", model_path)
        print("Training complete and model saved at", model_path)
    else:
        print("Training failed. See log for details.")

if __name__ == "__main__":
    main()
