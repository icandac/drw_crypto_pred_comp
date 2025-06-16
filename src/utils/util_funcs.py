import gc
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit

from src.utils.light_preprocess import Preprocessor


def save_submission(test_index, preds, out_path):
    """
    Save predictions to a CSV file for submission.
    Args:
        test_index (Index): Index of the test dataset.
        preds (np.ndarray): Predictions array.
        out_path (str): Path to save the submission file.
    """
    sub = pd.DataFrame({"ID": test_index, "prediction": preds})
    sub = sub.sort_values("ID")
    sub.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


def last_fraction(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    """
    Keep only the last `frac` share of rows (chronological tail).
    Args:
        df: pd.DataFrame
            Dataframe that this trimming will be applied.
        frac: float
            Fraction of the data will stay from the end, default = 0.1
    Returns:
        Trimmed input dataframe.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    start = int((1 - frac) * len(df))
    return df.iloc[start:].copy()


def variance_filter(
    df: pd.DataFrame, target_col: str = "label", thresh_ratio: float = 0.1
) -> pd.DataFrame:
    """
    Remove feature columns whose variance is < `thresh_ratio` x var(label).

    Args:
        df : DataFrame
            Training frame (label + features).
        thresh_ratio : float
            Keep features whose var >= thresh_ratio * var(label).

    Returns:
        DataFrame
    """
    label_var = df[target_col].var()
    thresh = label_var * thresh_ratio

    keep_cols = [target_col] + [
        c for c in df.columns if c != target_col and df[c].var() >= thresh
    ]
    pruned = df[keep_cols].copy()
    print(f"Variance filter kept {len(keep_cols)-1} of {df.shape[1]-1} columns")
    return pruned


def corr_cluster_select(
    df: pd.DataFrame, target: str = "label", thresh: float = 0.90
) -> pd.DataFrame:
    """
    Target-aware correlation-clustering filter.

    Keeps at most **one** feature from every group of highly-correlated
    columns (|corr| ≥ `thresh`).  For each group the survivor is the
    feature with the strongest absolute correlation to the target.
    Args:
        df : DataFrame
            Input frame that still contains `target`.
        thresh : float
            Absolute Pearson correlation threshold that triggers grouping
            (default 0.90).
    Returns:
        DataFrame
            Same rows, but with duplicates pruned.
    """
    feats = df.drop(columns=[target])
    y = df[target]

    corr = feats.astype("float32").corr().abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper = corr.where(mask)

    survivors: List[str] = []
    processed = set()  # columns already clustered

    for col in upper.columns:
        if col in processed:
            continue

        # all features strongly correlated with `col`
        cluster = upper.index[upper[col] >= thresh].tolist()
        cluster.append(col)

        processed.update(cluster)

        # pick the column with largest |corr| to the label
        best = feats[cluster].corrwith(y).abs().idxmax()
        survivors.append(best)

    # deduplicate while preserving order
    survivors = list(dict.fromkeys(survivors))

    kept_df = df[[target] + survivors].copy()
    print(
        f"Correlation filter kept {len(survivors)} "
        f"of {feats.shape[1]} columns (thresh={thresh})"
    )
    return kept_df


def compute_feature_importance(
    df: pd.DataFrame,
    target_col: str = "label",
    n_splits: int = 5,
    seed: int = 42,
    importance_type: str = "gain",  # "gain" or "split",
    var_ratio: float = 0.1,
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
    x = df.drop(columns=[target_col])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    imp_accum = pd.Series(0.0, index=x.columns)

    params = dict(
        objective="regression",
        learning_rate=0.1,
        num_leaves=256,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        seed=seed,
        verbose=-1,
    )

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(x), 1):
        print(f"Training fold {fold}/{n_splits} …", end="\r")

        x_tr = x.iloc[tr_idx].astype("float32")
        y_tr = y.iloc[tr_idx]
        x_val = x.iloc[val_idx].astype("float32")
        y_val = y.iloc[val_idx]

        tr_ds = lgb.Dataset(x_tr, y_tr, free_raw_data=False)
        val_ds = lgb.Dataset(x_val, y_val, reference=tr_ds, free_raw_data=False)

        model = lgb.train(
            params,
            tr_ds,
            num_boost_round=1000,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(200)],
        )

        imp_accum += pd.Series(
            model.feature_importance(importance_type=importance_type),
            index=x.columns,
            dtype="float64",
        )

        del model, tr_ds, val_ds, x_tr, x_val  # <── release memory now
        gc.collect()

    imp_df = (
        imp_accum.div(n_splits)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
    )

    return imp_df


def crossval_pearson(x, y, n_splits=5, seed=42) -> float:
    """Return mean Pearson over TimeSeriesSplit folds."""
    from scipy.stats import pearsonr
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, val_idx in tscv.split(x):
        x_tr = x.iloc[tr_idx].astype("float32")
        y_tr = y.iloc[tr_idx]
        x_val = x.iloc[val_idx].astype("float32")
        y_val = y.iloc[val_idx]

        m = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.1,
            num_leaves=256,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            n_estimators=600,
            random_state=seed,
        )
        m.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=0),
            ],
        )
        preds = m.predict(x_val, num_iteration=m.best_iteration_)
        scores.append(pearsonr(y_val, preds)[0])

        del m, x_tr, x_val
        gc.collect()

    return float(np.mean(scores))


def optimize_topn_features(
    train_raw: pd.DataFrame,
    imp_df: pd.DataFrame,
    top_min: int = 10,
    top_max: int = 700,
    n_trials: int = 30,
    timeout: int = 1800,
    seed: int = 42,
) -> Tuple[int, float]:
    """
    Use Optuna to pick the optimal `top_n` (# of most-important features
    to expand with lags/rolls).

    Args:
        train_raw : DataFrame
            Original training frame (features + 'label').
        imp_df : DataFrame
            Output of `compute_feature_importance`, already sorted desc.
        top_min, top_max : int
            Search range for `top_n`.
        n_trials : int
            Max Optuna trials.
        timeout : int
            Seconds; stop earlier if reached.
        seed : int
            Reproducibility.

    Returns:
        best_n : int
            Chosen cut-off.
        best_cv : float
            Mean CV Pearson achieved with that cut-off.
    """

    def objective(trial: optuna.Trial) -> float:
        # Suggest a cut-off (log scale samples small & large)
        top_n = trial.suggest_int("top_n", top_min, top_max, log=True)
        cols = imp_df.head(top_n)["feature"].tolist()

        # Build pre-processor with those columns
        pre = Preprocessor(
            lag_steps=[1, 5, 30],
            rolling_windows=[5, 30, 120],
            clip_quantiles=(0.001, 0.999),
            expand_cols=cols,
            aggregate=None,
        )
        train = pre.fit_transform(train_raw)
        x = train.drop(columns=["label"]).astype("float32")
        y = train["label"]

        # Evaluate
        pear = crossval_pearson(x, y, n_splits=5, seed=seed)
        trial.set_user_attr("pearson", pear)

        del train, x, y, pre, cols
        gc.collect()

        return -pear

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
    )

    best_n = study.best_params["top_n"]
    best_cv = -study.best_value
    print(f"Optuna picked Top-{best_n} with CV Pearson {best_cv:.5f}")
    return best_n, best_cv


def oof_pearson(x, y, n_splits=5, seed=42):
    """
    Return the single Pearson correlation across the entire training
    set, using out-of-fold predictions (TimeSeriesSplit).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = np.full(len(y), np.nan, dtype=np.float32)

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

    for tr_idx, val_idx in tscv.split(x):
        x_tr = x.iloc[tr_idx].astype("float32")
        y_tr = y.iloc[tr_idx]
        x_val = x.iloc[val_idx].astype("float32")

        model = lgb.LGBMRegressor(**params, n_estimators=10_000)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y.iloc[val_idx])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_pred[val_idx] = model.predict(x_val, num_iteration=model.best_iteration_)

        # housekeeping – free RAM
        del model, x_tr, x_val
        gc.collect()

    mask = ~np.isnan(oof_pred)  # rows actually predicted

    return pearsonr(y.iloc[mask], oof_pred[mask])[0]
