import os
from pathlib import Path
import sys
import joblib
import warnings
import numpy as np
import pandas as pd

# Local imports
src_path = Path("../src").resolve()
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from loader import DataLoader

class FeatureEngineer:
    def __init__(self, lags, rolling_windows):
        self.lags = lags
        self.rolling_windows = rolling_windows

    def add_features(self, X):
        X_new = X.copy()

        # Add lag features
        for lag in self.lags:
            lagged = X.shift(lag).add_suffix(f"_lag{lag}")
            X_new = pd.concat([X_new, lagged], axis=1)

        # Add rolling features
        for window in self.rolling_windows:
            rolled = X.rolling(window=window).mean().add_suffix(f"_roll{window}")
            X_new = pd.concat([X_new, rolled], axis=1)

        X_new = X_new.fillna(0)
        return X_new

def make_filename(model_name, params):
    parts = [model_name]
    # Pick a few params to include in the name (max 3 for brevity)
    for k, v in list(params.items())[:3]:
        # clean keys and values for filename safety
        k_clean = k.replace('__', '-').replace('_', '')
        v_clean = str(v).replace('.', 'p')
        parts.append(f"{k_clean}{v_clean}")
    filename = "_".join(parts) + ".csv"
    return filename

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent of src folder
    data_path = os.path.join(base_dir, 'rawdata', 'test.parquet')
    model_path = os.path.join(base_dir, 'models', 'current.joblib')

    assert os.path.exists(data_path), f"File not found: {data_path}"
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    selected_columns = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", 
        "X674","X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", 
        "X168", "X612","bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume",
        "label"]
    
    # Load test data
    df = pd.read_parquet(data_path, columns=selected_columns)

    loaded = joblib.load("models/current.joblib")
    model = loaded['model']
    expected_features = loaded['feature_names']
    model_name = loaded['model_name']
    best_params = loaded['best_params']

    X_raw = df.drop(columns=['label'])
    lags = [1, 2, 3]
    rolling_windows = [3]

    fe = FeatureEngineer(lags=lags, rolling_windows=rolling_windows)
    X_new = fe.add_features(X_raw)
    X_new = X_new[expected_features]

    preds = model.predict(X_new)

    # Convert predictions to DataFrame with ID
    pred_df = pd.DataFrame({
        'ID': X_new.index,
        'prediction': preds
    })

    filename = make_filename(model_name, best_params)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    filepath = os.path.join(output_dir, filename)
    pred_df.to_csv(filepath, index=False)

    print(f"Predictions saved to {filepath}")
