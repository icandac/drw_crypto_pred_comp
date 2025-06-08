import warnings

import lightgbm as lgb
import pandas as pd
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


def train_and_pick_best(X, y, n_splits=5):
    """
    Train LightGBM models using Time Series Cross-Validation and pick the
    best model based on Pearson correlation.
    Args:
        X (DataFrame): Feature DataFrame.
        y (Series): Target variable.
        n_splits (int): Number of splits for Time Series Cross-Validation.
    Returns:
        LightGBM model: The best model based on validation Pearson correlation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_model = None
    best_corr = -9
    best_fold_id = None

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

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
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
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=200),
            ],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        corr, _ = pearsonr(y_val, val_pred)
        print(f"Fold {fold}/{n_splits} – Pearson = {corr:.5f}")

        if corr > best_corr:
            best_corr, best_model, best_fold_id = corr, model, fold

    print(f"✓ Best fold: {best_fold_id}  (Pearson {best_corr:.5f})")
    return best_model


def predict(best_model, X_test):
    """
    Make predictions using the trained models.
    Args:
        models (list): List of trained LightGBM models.
        X_test (DataFrame): Test feature DataFrame.
    Returns:
        np.ndarray: Average predictions from all models.
    """
    return best_model.predict(X_test, num_iteration=best_model.best_iteration)


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
    best_model = train_and_pick_best(X_train, y_train, n_splits=5)
    preds = predict(best_model, X_test)
    save_submission(test.index, preds)


if __name__ == "__main__":
    main()
