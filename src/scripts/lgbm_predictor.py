import gc
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit

from src.utils.light_preprocess import Preprocessor
from src.utils.util_funcs import (
    compute_feature_importance,
    save_submission,
)

gc.collect()
np.seterr(invalid="ignore")
warnings.filterwarnings("ignore", category=UserWarning)

seed = 42
data_folder = "data/"
output_folder = "outputs/"
cols_to_expand = ["bid_qty", "ask_qty", "volume"]


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
    # print(len(train), len(test))

    # train = train[int(len(train)):]
    # test = test[int(len(test)):]
    # print(len(train), len(test))
    return train, test


def feature_col_filter(train, test):
    """
    Preprocess the training and test datasets.
    Args:
        train (DataFrame): Training dataset.
        test (DataFrame): Test dataset.
    Returns:
        tuple: (x_train DataFrame, y_train Series, x_test DataFrame)
    """
    # Ensure timestamp index is sorted (important for TSCV)
    if isinstance(train.index, pd.DatetimeIndex):
        train = train.sort_index()

    # Align feature columns between train & test
    feature_cols = [c for c in train.columns if c != "label"]
    x_train = train[feature_cols]
    y_train = train["label"]
    x_test = test[feature_cols]
    return x_train, y_train, x_test


def train_and_pick_best(x, y, n_splits=5):
    """
    Train LightGBM models using Time Series Cross-Validation and pick the
    best model based on Pearson correlation.
    Args:
        x (DataFrame): Feature DataFrame.
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

    for fold, (train_idx, val_idx) in enumerate(tscv.split(x), 1):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        tr_ds = lgb.Dataset(x_tr, y_tr, free_raw_data=False)
        val_ds = lgb.Dataset(x_val, y_val, reference=tr_ds, free_raw_data=False)

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

        val_pred = model.predict(x_val, num_iteration=model.best_iteration)
        corr, _ = pearsonr(y_val, val_pred)
        print(f"Fold {fold}/{n_splits} - Pearson = {corr:.5f}")

        if corr > best_corr:
            best_corr, best_model, best_fold_id = corr, model, fold

    print(f"Best fold: {best_fold_id}  (Pearson {best_corr:.5f})")
    return best_model


def predict(best_model, x_test):
    """
    Make predictions using the trained models.
    Args:
        models (list): List of trained LightGBM models.
        x_test (DataFrame): Test feature DataFrame.
    Returns:
        np.ndarray: Average predictions from all models.
    """
    return best_model.predict(x_test, num_iteration=best_model.best_iteration)


def lgbm_runner():
    train_raw, test_raw = load_data()
    print("Computing feature importance …")

    imp_df = compute_feature_importance(
        train_raw, target_col="label", n_splits=5, var_ratio=0.1
    )
    imp_df.to_csv(f"{output_folder}feature_importance.csv", index=False)
    # # Open the following row to auto-search the best Top-N
    # best_n, _ = optimize_topn_features(
    #     train_raw, imp_df, top_min=10, top_max=700, n_trials=30, timeout=1800
    # )
    # cols_to_expand = imp_df.head(best_n)["feature"].tolist()
    # print(f"Top {len(best_n)} features:\n", imp_df.head(len(best_n)))
    top_n = 48
    print(f"Top {top_n} features:\n", imp_df.head(int(top_n)))
    # pick N best columns for lag / roll
    cols_to_expand = imp_df.head(top_n)["feature"].tolist()
    train_reduced = train_raw[["label", *cols_to_expand]].copy()
    test_reduced = test_raw[cols_to_expand].copy()

    pre = Preprocessor(
        lag_steps=[1, 5, 30],
        rolling_windows=[5, 30, 60, 120, 1440],
        clip_quantiles=(0.001, 0.999),
        expand_cols=cols_to_expand,
        aggregate=None,
    )

    print("Fitting preprocessor...")
    train = pre.fit_transform(train_reduced)
    test = pre.transform(test_reduced)

    x_train, y_train, x_test = feature_col_filter(train, test)

    print("Training model...")
    best_model = train_and_pick_best(x_train, y_train, n_splits=5)

    print("Making predictions...")
    preds = predict(best_model, x_test)

    save_submission(test.index, preds, out_path=f"{output_folder}submission_lgbm.csv")
