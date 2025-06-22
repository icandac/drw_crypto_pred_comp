import os
from pathlib import Path
import sys
import joblib
import warnings
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge


# Local imports
src_path = Path("../src").resolve()
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from loader import DataLoader

class LagFeatureEngineer:
    def __init__(self, lags=[1, 2, 3], rolling_windows=[3]):
        self.lags = lags
        self.rolling_windows = rolling_windows

    def add_lag_features(self, df):
        df_lagged = df.copy()
        for lag in self.lags:
            lagged = df.shift(lag).add_suffix(f'_lag{lag}')
            df_lagged = pd.concat([df_lagged, lagged], axis=1)
        return df_lagged

    def add_lag_and_rolling_features(self, df):
        df_features = df.copy()
        for lag in self.lags:
            lagged = df.shift(lag).add_suffix(f'_lag{lag}')
            df_features = pd.concat([df_features, lagged], axis=1)
        for window in self.rolling_windows:
            rolling = df.rolling(window=window).mean().add_suffix(f'_roll{window}')
            df_features = pd.concat([df_features, rolling], axis=1)
        return df_features

    def transform(self, X):
        # Apply both lags and rolling windows
        df_transformed = self.add_lag_and_rolling_features(X)
        df_transformed = df_transformed.dropna()
        return df_transformed
 


# Safe Pearson scorer
def safe_pearsonr(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y_true = np.ravel(y_true)
            y_pred = np.ravel(y_pred)
            r, _ = pearsonr(y_true, y_pred)
            return 0 if np.isnan(r) else r
        except Exception:
            return 0

pearson_scorer = make_scorer(safe_pearsonr, greater_is_better=True)

class ModelTrainer:
    def __init__(self, model, param_grid=None, lags=[1,2,3], rolling_windows=[3], n_splits=5):
        self.model = model
        self.param_grid = param_grid or {}
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.n_splits = n_splits
        
        self.best_model = None
        self.best_score = None
        self.best_params = {}

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

        X_new = X_new.dropna()
        return X_new

    def prepare_features(self, X, y):
        X_feat = self.add_features(X)
        y_feat = y.loc[X_feat.index]
        return X_feat, y_feat

    def grid_search(self, X, y):
        X_feat, y_feat = self.prepare_features(X, y)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model)
        ])

        if self.param_grid:
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grid,
                cv=TimeSeriesSplit(n_splits=self.n_splits),
                scoring=pearson_scorer,
                n_jobs=-1,
                verbose=2
            )
            grid.fit(X_feat, y_feat)
            self.best_model = grid.best_estimator_
            self.best_score = grid.best_score_
            self.best_params = grid.best_params_
        else:
            # No param grid: manually do CV scoring
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []

            for train_idx, test_idx in tscv.split(X_feat):
                X_train, X_test = X_feat.iloc[train_idx], X_feat.iloc[test_idx]
                y_train, y_test = y_feat.iloc[train_idx], y_feat.iloc[test_idx]
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                score = safe_pearsonr(y_test, y_pred)
                scores.append(score)

            self.best_model = pipeline.fit(X_feat, y_feat)
            self.best_score = np.mean(scores)
            self.best_params = {}

    def retrain_on_full_data(self, X, y):
        X_feat, y_feat = self.prepare_features(X, y)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model.__class__(**self._extract_model_params()))
        ])
        self.best_model = pipeline.fit(X_feat, y_feat)

    def save(self, path):
        joblib.dump({
            'model': self.best_model,
            'lags': self.lags,
            'rolling_windows': self.rolling_windows
        }, path)

    def _extract_model_params(self):
        """
        Convert self.best_params (from GridSearchCV) to plain model kwargs.
        Example: 'model__alpha' -> 'alpha'
        """
        return {k.split("__")[1]: v for k, v in self.best_params.items()} if self.best_params else {}


if __name__ == "__main__":
    # Load data
    #X_top50 = pd.read_parquet('rawdata/X_top50.parquet')
    #y_df = pd.read_parquet('rawdata/y.parquet')

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent of src folder
    data_path = os.path.join(base_dir, 'rawdata', 'train.parquet')
    assert os.path.exists(data_path), f"File not found: {data_path}"

    loader = DataLoader()
    selected_columns = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", 
        "X674","X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", 
        "X168", "X612","bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume",
        "label"]
    df = loader.load_and_preprocess(data_path, start_date="2023-12-01", columns=selected_columns)
    target_column = 'label'
    y_df = df[target_column]
    X_top50 = df.drop(columns=[target_column])

    y_df = y_df.squeeze()

    model_configs = [
        {
            'name': 'ElasticNet',
            'model': ElasticNet(random_state=42, max_iter=10000),
            'param_grid': {
                'model__alpha': [0.1],
                'model__l1_ratio': [0.35, 0.5]
            }
        },
        {
            'name': 'LinearRegression',
            'model': LinearRegression(),
            'param_grid': {}
        },
        {
            'name': 'Ridge',
            'model': Ridge(random_state=42, max_iter=10000),
            'param_grid': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        {
            'name': 'LightGBM',
            'model': lgb.LGBMRegressor(random_state=42, n_jobs=4, force_row_wise=True),
            'param_grid': {
                'model__num_leaves': [31],
                'model__learning_rate': [0.05, 0.1, 0.2],
            }
        }
    ]
    
    results = []

    for config in model_configs:
        print(f"Running model: {config['name']}")
        trainer = ModelTrainer(
            model=config['model'],
            param_grid=config['param_grid'],
            lags=[1,2,3],
            rolling_windows=[3],
            n_splits=5
        )
        trainer.grid_search(X_top50, y_df)
        X_feat, y_feat = trainer.prepare_features(X_top50, y_df)
        feature_names = X_feat.columns.tolist()

        print(f"{config['name']} mean Pearson r: {trainer.best_score:.4f}")

        results.append({
            'model': config['name'],
            'score': trainer.best_score,
            'best_params': trainer.best_params,
            'trainer': trainer
        })

    best_result = max(results, key=lambda x: x['score'])
    print("Best model overall:", best_result['model'], best_result['score'])
    joblib.dump(
        {
            'model': best_result['trainer'].best_model,
            'feature_names': feature_names,
            'model_name': best_result['model'],
            'best_params': best_result['best_params']
        },
        "models/current.joblib"
    )    

    # Print summary
    for res in results:
        print(res)

    #print(X_top50.columns)

