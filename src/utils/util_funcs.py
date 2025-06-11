import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def save_submission(test_index, preds, out_path):
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


def compute_feature_importance(
    df: pd.DataFrame,
    target_col: str = "label",
    n_splits: int = 5,
    seed: int = 42,
    importance_type: str = "gain",  # "gain" or "split"
) -> pd.DataFrame:
    """
    Train LightGBM on each fold, average feature importances, and
    return a ranked table.

    Args:
        df : DataFrame
            Raw training frame (timestamp index, features + label).
        target_col : str
            Column name of the regression target.
        n_splits : int
            TimeSeriesSplit folds.
        importance_type : str
            "gain" - total gain of splits (default, more robust)
            "split" - number of times the feature is used in splits.

    Returns:
        DataFrame with columns [feature, importance] sorted descending.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    imp_accum = pd.Series(0.0, index=X.columns)

    params = dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=256,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        seed=seed,
        verbose=-1,
    )

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"⏳  Training fold {fold}/{n_splits} …", end="\r")

        tr_ds = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
        val_ds = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx], reference=tr_ds)

        model = lgb.train(
            params,
            tr_ds,
            num_boost_round=4000,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(200)],
        )

        imp_accum += pd.Series(
            model.feature_importance(importance_type=importance_type),
            index=X.columns,
            dtype="float64",
        )

    imp_df = (
        imp_accum.div(n_splits)  # average across folds
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
    )

    return imp_df
