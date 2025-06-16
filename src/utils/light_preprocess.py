from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class Preprocessor:
    """
    Memory-aware feature engineer for large crypto datasets.
    This class handles:
    - Lag features
    - Rolling statistics
    - Clipping outliers
    - NaN handling
    - Adding order book imbalance feature
    - Optionally expanding columns for lags/rolls
    - Optionally aggregating lags/rolls (mean, median)
    - Optionally adding imbalance feature from order book quantities
    """

    def __init__(
        self,
        lag_steps: List[int],
        rolling_windows: List[int],
        clip_quantiles: Tuple[float, float] = (0.005, 0.995),
        expand_cols: Optional[List[str]] = None,  # which columns get lags/rolls
        aggregate: Optional[str] = None,  # 'mean','median' or None
        add_imbalance: bool = True,
        orderbook_cols: Tuple[str, str] = ("bid_qty", "ask_qty"),
    ):
        self.lag_steps = list(lag_steps)
        self.rolling_windows = list(rolling_windows)
        self.clip_quantiles = clip_quantiles
        self.expand_cols = expand_cols  # if None → all numeric
        self.aggregate = aggregate
        self.add_imbalance = add_imbalance
        self.orderbook_cols = orderbook_cols

        self._clip_bounds: Dict[str, Tuple[float, float]] = {}
        self._medians: Dict[str, float] = {}
        self._base_cols: List[str] = []

    def fit(self, df: pd.DataFrame):
        feats = [c for c in df.columns if c != "label"]
        self._base_cols = feats
        qlo, qhi = self.clip_quantiles
        q = df[feats].quantile([qlo, qhi])
        self._clip_bounds = dict(zip(feats, zip(q.loc[qlo], q.loc[qhi])))
        self._medians = df[feats].median().to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # start with base features in float32 to halve RAM
        out = df.copy()
        out[self._base_cols] = out[self._base_cols].astype("float32")

        # --- clipping & NaN fill vectorised ---
        for c, (lo, hi) in self._clip_bounds.items():
            col = out[c].clip(lo, hi)
            col = col.ffill().bfill().fillna(self._medians[c])
            out[c] = col

        expand = self.expand_cols or self._base_cols
        pieces = [out]  # list of frames to concat later

        # --- lags ---
        for lag in self.lag_steps:
            shifted = out[expand].shift(lag)
            if self.aggregate:
                pieces.append(
                    shifted.aggregate(self.aggregate, axis=1)
                    .rename(f"lag{lag}_{self.aggregate}")
                    .to_frame()
                    .astype("float32")
                )
            else:
                shifted.columns = [f"{c}_lag{lag}" for c in shifted]
                pieces.append(shifted.astype("float32"))
                del shifted

        # --- rolling ---
        for win in self.rolling_windows:
            roll = out[expand].rolling(win, min_periods=1)
            if self.aggregate:
                pieces.append(
                    roll.mean()
                    .mean(axis=1)
                    .rename(f"roll_mean{win}")
                    .to_frame()
                    .astype("float32")
                )
                pieces.append(
                    roll.std()
                    .mean(axis=1)
                    .rename(f"roll_std{win}")
                    .to_frame()
                    .astype("float32")
                )
            else:
                pieces.append(roll.mean().add_suffix(f"_mean{win}").astype("float32"))
                pieces.append(roll.std().add_suffix(f"_std{win}").astype("float32"))

        # --- imbalance ---
        if self.add_imbalance and all(c in out.columns for c in self.orderbook_cols):
            bid, ask = self.orderbook_cols
            denom = (out[bid] + out[ask]).replace(0.0, np.nan)
            imb = ((out[bid] - out[ask]) / denom).fillna(0).astype("float32")
            pieces.append(imb.rename("orderbook_imbalance").to_frame())

        # concat ONCE
        final = pd.concat(pieces, axis=1)
        return final

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
