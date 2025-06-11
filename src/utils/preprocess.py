from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class Preprocessor:
    """
    A class for preprocessing time series data with feature engineering.
    This class handles:
    - Lag features
    - Rolling statistics
    - Clipping outliers
    - NaN handling
    - Adding time-based features
    - Adding order book imbalance feature
    """

    def __init__(
        self,
        lag_steps: List[int],
        rolling_windows: List[int],
        clip_quantiles: Tuple[float, float] = (0.005, 0.995),
        add_time_features: bool = True,
        add_imbalance: bool = True,
        orderbook_cols: tuple[str, str] = ("bid_qty", "ask_qty"),
    ):
        self.lag_steps = list(lag_steps)
        self.rolling_windows = list(rolling_windows)
        self.clip_quantiles = clip_quantiles
        self.add_time_features = add_time_features
        self.add_imbalance = add_imbalance
        self.orderbook_cols = orderbook_cols

        # learned attributes
        self._clip_bounds: Dict[str, Tuple[float, float]] = {}
        self._medians: Dict[str, float] = {}
        self._feature_cols_: List[str] = []

    def fit(self, df: pd.DataFrame):
        quant_lo, quant_hi = self.clip_quantiles
        features = [c for c in df.columns if c != "label"]

        # Compute clipping bounds & medians
        for col in features:
            lo, hi = df[col].quantile([quant_lo, quant_hi])
            self._clip_bounds[col] = (lo, hi)
            self._medians[col] = df[col].median()

        self._feature_cols_ = features
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Clip outliers column‑wise
        for col in self._feature_cols_:
            lo, hi = self._clip_bounds[col]
            out[col] = out[col].clip(lo, hi)

        # NaN handling
        out[self._feature_cols_] = (
            out[self._feature_cols_].fillna(method="ffill").fillna(method="bfill")
        )
        for col in self._feature_cols_:
            out[col] = out[col].fillna(self._medians[col])

        # Feature engineering
        out = self._add_lag_features(out)
        out = self._add_roll_features(out)
        if self.add_imbalance and all(c in out.columns for c in self.orderbook_cols):
            out = self._add_imbalance_feature(out)
        if self.add_time_features and isinstance(out.index, pd.DatetimeIndex):
            out = self._add_time_features(out)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ===== helper methods =====================================
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ldf = df.copy()
        for lag in self.lag_steps:
            shifted = df[self._feature_cols_].shift(lag)
            shifted.columns = [f"{c}_lag{lag}" for c in shifted.columns]
            ldf = ldf.join(shifted)
        return ldf

    def _add_roll_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rdf = df.copy()
        for win in self.rolling_windows:
            roll = df[self._feature_cols_].rolling(window=win, min_periods=1)
            rdf = rdf.join(roll.mean().add_suffix(f"_mean{win}"))
            rdf = rdf.join(roll.std().add_suffix(f"_std{win}"))
            rdf = rdf.join(roll.min().add_suffix(f"_min{win}"))
            rdf = rdf.join(roll.max().add_suffix(f"_max{win}"))
        return rdf

    def _add_imbalance_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        bid, ask = self.orderbook_cols
        denom = (df[bid] + df[ask]).replace(0, np.nan)
        df["orderbook_imbalance"] = (df[bid] - df[ask]) / denom
        df["orderbook_imbalance"] = df["orderbook_imbalance"].fillna(0.0)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dti = df.index
        df["hour"] = dti.hour
        df["dayofweek"] = dti.dayofweek
        df["month"] = dti.month
        return df.astype({"hour": "uint8", "dayofweek": "uint8", "month": "uint8"})
