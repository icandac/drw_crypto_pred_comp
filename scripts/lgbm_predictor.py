import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

seed = 42
data_folder = "data/"
output_folder = "outputs/"


def load_data(
    train_path=f"{data_folder}train.parquet", test_path=f"{data_folder}test.parquet"
):
    """
    Load training and test datasets from parquet files.
    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.
    Returns:
        tuple: (train DataFrame, test DataFrame)
    """
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    return train, test


def preprocess(train, test):
    """
    Preprocess the training and test datasets.
    Args:
        train (DataFrame): Training dataset.
        test (DataFrame): Test dataset.
    Returns:
        tuple: (X_train DataFrame, y_train Series, X_test DataFrame)
    """
    # Ensure timestamp index is sorted (important for TSCV)
    if isinstance(train.index, pd.DatetimeIndex):
        train = train.sort_index()

    # Align feature columns between train & test
    feature_cols = [c for c in train.columns if c != "label"]
    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    return X_train, y_train, X_test


def train_models(X, y, n_splits=5):
    """
    Train LightGBM models using Time Series Cross-Validation.
    Args:
        X (DataFrame): Feature DataFrame.
        y (Series): Target variable.
        n_splits (int): Number of splits for Time Series Cross-Validation.
    Returns:
        list: List of trained LightGBM models.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    scores = []

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 256,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(period=100),
    ]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        tr_ds = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
        val_ds = lgb.Dataset(X_val, y_val, reference=tr_ds, free_raw_data=False)

        model = lgb.train(
            lgb_params,
            tr_ds,
            num_boost_round=10_000,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=callbacks,
        )

        # Validation Pearson correlation
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        corr, _ = pearsonr(y_val, val_pred)
        scores.append(corr)
        print(f"Fold {fold+1}/{n_splits}  –  Pearson: {corr:.5f}")

        models.append(model)
        dump(model, f"{output_folder}model_fold{fold+1}.joblib")

    print(f"Average Pearson over {n_splits} folds: {np.mean(scores):.5f}")
    return models


def predict(models, X_test):
    """
    Make predictions using the trained models.
    Args:
        models (list): List of trained LightGBM models.
        X_test (DataFrame): Test feature DataFrame.
    Returns:
        np.ndarray: Average predictions from all models.
    """
    # Average the predictions from each fold
    preds = np.mean(
        [m.predict(X_test, num_iteration=m.best_iteration) for m in models], axis=0
    )
    return preds


def save_submission(test_index, preds, out_path=f"{output_folder}submission_lgbm.csv"):
    """
    Save predictions to a CSV file for submission.
    Args:
        test_index (Index): Index of the test dataset.
        preds (np.ndarray): Predictions array.
        out_path (str): Path to save the submission file.
    """
    sub = pd.DataFrame({"ID": test_index, "label": preds})
    sub.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


def main():
    train, test = load_data()
    X_train, y_train, X_test = preprocess(train, test)
    models = train_models(X_train, y_train, n_splits=5)
    preds = predict(models, X_test)
    save_submission(test.index, preds)


if __name__ == "__main__":
    main()
